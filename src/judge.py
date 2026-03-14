from __future__ import annotations

import json
import logging
import os
from string import Template
from typing import Annotated, Literal, Optional

from typing_extensions import TypedDict

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

# from agents.tools.pdf_to_txt_transformer import pdf_to_text

from src.llm import LLM
from pypdf import PdfReader


logger = logging.getLogger(__name__)

load_dotenv()


class State(TypedDict):
    messages: Annotated[list, add_messages]
    pdf_path: Optional[str]
    sentencia_texto: Optional[str]
    facts_json: Optional[str]
    eval_json: Optional[str]
    output_json: Optional[str]


JUDGE_PROMPT = """Eres un asistente en español que actúa como un "juez simple"
NO VINCULANTE para análisis de sentencias de divorcio.
Tu tarea es LEER el texto aportado (puede estar incompleto) y producir salidas
ESTRICTAMENTE en JSON según el esquema pedido.

Reglas:
- No inventes datos. Si falta, usa null, listas vacías o "desconocido" cuando corresponda.
- Mantén un tono formal, neutral y jurídico (sin dar asesoría legal personalizada).
- Si detectas jurisdicción, adapta términos, pero sin citar artículos salvo que ya estén en el texto.
- No incluyas texto fuera del JSON.
"""


class ExtractedFacts(BaseModel):

    jurisdiccion: Optional[str] = Field(
        default=None,
        description="País/CCAA/ciudad si aparece (p. ej. España, CDMX).",
    )
    organo_judicial: Optional[str] = Field(
        default=None, description="Juzgado/tribunal y número si aparece."
    )
    tipo_procedimiento: Optional[str] = Field(
        default=None,
        description=(
            "Mutuo acuerdo / contencioso / divorcio incausado / separación, etc."
        ),
    )
    fecha: Optional[str] = Field(
        default=None, description="Fecha de la sentencia si aparece."
    )
    partes: list[str] = Field(
        default_factory=list, description="Nombres/identificadores de las partes."
    )
    hijos_menores: Optional[bool] = Field(
        default=None, description="Si hay hijos menores."
    )
    hijos_detalle: list[str] = Field(
        default_factory=list, description="Edades/nombres si aparecen."
    )
    custodia: Optional[str] = Field(
        default=None, description="Monoparental/compartida/no aplica."
    )
    regimen_visitas: Optional[str] = Field(
        default=None, description="Régimen de visitas si aplica."
    )
    pension_alimentos: Optional[str] = Field(
        default=None, description="Importe/periodicidad/actualización si aparece."
    )
    pension_compensatoria: Optional[str] = Field(
        default=None, description="Si aparece y condiciones."
    )
    vivienda_familiar: Optional[str] = Field(
        default=None, description="Atribución de uso si aparece."
    )
    regimen_economico: Optional[str] = Field(
        default=None, description="Gananciales/separación/bienes, etc."
    )
    liquidacion_regimen: Optional[str] = Field(
        default=None, description="Si se aborda liquidación/partición."
    )
    medidas_proteccion: Optional[str] = Field(
        default=None, description="Órdenes de protección/violencia si se menciona."
    )
    fallo_literal: Optional[str] = Field(
        default=None, description="Extracto del fallo si está claramente identificado."
    )


class Evaluation(BaseModel):
    coherencia_interna: Literal["alta", "media", "baja"] = Field(
        description="Qué tan consistente/coherente parece el texto."
    )
    motivacion_suficiente: Literal["sí", "parcial", "no"] = Field(
        description=(
            "Si parece haber motivación/razonamiento suficiente (a simple vista)."
        )
    )
    puntos_faltantes: list[str] = Field(
        default_factory=list,
        description="Elementos típicos que no aparecen o quedan ambiguos.",
    )
    banderas_rojas: list[str] = Field(
        default_factory=list,
        description="Posibles problemas: contradicciones, falta de cuantías, plazos, etc.",
    )
    recomendaciones: list[str] = Field(
        default_factory=list,
        description=(
            "Sugerencias prácticas (p. ej. pedir aclaración, revisar ejecución, etc.)."
        ),
    )


class JudgeOutput(BaseModel):
    resumen: str = Field(
        description="Resumen breve y neutral de la sentencia.")
    hechos_extraidos: ExtractedFacts
    evaluacion: Evaluation
    fallo_simulado: str = Field(
        description="Un 'fallo' redactado (no vinculante) con base en el texto."
    )
    disclaimer: str = Field(
        description="Aclaración de que no es asesoría legal y depende de jurisdicción."
    )


class DivorceJudge:

    def __init__(self, llm: LLM) -> None:
        self._llm = llm

    def analyze_case(
        self,
        divorce_case_file_path: str
    ):
        # Create a graph that tells the judge the "steps" to do.
        app = self._create_graph()

        # Judge the case and return a final response
        result = app.invoke(
            {
                "messages": [],
                "pdf_path":  divorce_case_file_path,
                "sentencia_texto": None,
                "facts_json": None,
                "eval_json": None,
                "output_json": None,
            }
        )
        return result

    def _transform_pdf_to_text(self, state: State) -> str:
        pdf_file_path = state["pdf_path"]
        pdf_reader = PdfReader(pdf_file_path)
        text = ""

        for index, each_page in enumerate(pdf_reader.pages):
            logger.info("Page [%s]: parsing from PDF page to text", index)
            text += f"---- START PAGE {index} ----"
            text += each_page.extract_text() + "\n"
            text += f"---- END PAGE {index} ----"

        return {
            "sentencia_texto": text
        }

    def _create_graph(self,):
        graph = StateGraph(State)

        # Nodes of the graph
        graph.add_node("transform_pdf_to_text", self._transform_pdf_to_text)
        graph.add_node("extract_facts", self._extract_facts)
        graph.add_node("evaluate", self._evaluate)
        graph.add_node("render_output", self.render_output)

        # Conections between nodes
        graph.add_edge(START, "transform_pdf_to_text")
        graph.add_edge("transform_pdf_to_text", "extract_facts")
        graph.add_edge("extract_facts", "evaluate")
        graph.add_edge("evaluate", "render_output")
        graph.add_edge("render_output", END)

        return graph.compile()

    def _extract_facts(
        self,
        state: State
    ):
        logger.info("Extracting facts from sentencia...")
        extract_facts_prompt = """Extrae HECHOS de la sentencia de divorcio.
Devuelve SOLO JSON compatible con este modelo:
{
  "jurisdiccion": string|null,
  "organo_judicial": string|null,
  "tipo_procedimiento": string|null,
  "fecha": string|null,
  "partes": [string],
  "hijos_menores": boolean|null,
  "hijos_detalle": [string],
  "custodia": string|null,
  "regimen_visitas": string|null,
  "pension_alimentos": string|null,
  "pension_compensatoria": string|null,
  "vivienda_familiar": string|null,
  "regimen_economico": string|null,
  "liquidacion_regimen": string|null,
  "medidas_proteccion": string|null,
  "fallo_literal": string|null
}
Texto de la sentencia:
<<<
$sentencia
>>>
"""
        # Get the sentence as text
        sentencia = state.get("sentencia_texto")

        # Create the instruction for the LLM
        instruction = [
            SystemMessage(content=JUDGE_PROMPT),
            HumanMessage(
                content=self._tpl(extract_facts_prompt, sentencia=sentencia)
            ),
        ]

        # Call the LLM with the instruction to get a formatted output from
        # the long pdf
        resp = self._llm.call(instruction)

        # Get the response from the LLM
        facts_raw = resp.content

        # Make sure the response is with the format we expect
        data = self._robust_json_load(facts_raw)
        facts = ExtractedFacts.model_validate(data)

        # Add the formatted sentence to the state of the graph
        return {
            "facts_json": facts.model_dump_json(ensure_ascii=False)
        }

    def _evaluate(self, state: State):
        evaluate_judgment_prompt = """Evalúa la calidad y completitud (a simple vista) de la sentencia,
SIN inventar. Devuelve SOLO JSON:
{
  "coherencia_interna": "alta"|"media"|"baja",
  "motivacion_suficiente": "sí"|"parcial"|"no",
  "puntos_faltantes": [string],
  "banderas_rojas": [string],
  "recomendaciones": [string]
}

Hechos extraídos (JSON):
$facts_json

Texto original:
<<<
$sentencia
>>>
"""
        msg = [
            SystemMessage(content=JUDGE_PROMPT),
            HumanMessage(
                content=self._tpl(
                    evaluate_judgment_prompt,
                    facts_json=state.get("facts_json"),
                    sentencia=state.get("sentencia_texto"),
                ),
            ),
        ]
        # Call the LLM with the instruction to evaluate the judgment
        resp = self._llm.call(msg)

        # Get the reponse
        eval_raw = resp.content

        # Make sure the response is formatted as expected
        data = self._robust_json_load(eval_raw)
        evaluation = Evaluation.model_validate(data)

        # Add this evaluation to the state of the graph
        return {
            "eval_json": evaluation.model_dump_json(ensure_ascii=False)
        }

    def _robust_json_load(self, s: str) -> object:
        # Try direct JSON first
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            pass
        # Try to extract the first JSON object in the text
        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(s[start: end + 1].replace("\ufeff", ""))

    def _tpl(self, s: str, **kwargs: str) -> str:
        return Template(s).safe_substitute(**kwargs)

    def render_output(self, state: State):
        logger.info("Creating a formatted output from the judge...")
        judge_output_format_prompt = """Con los HECHOS y la EVALUACIÓN, redacta una salida final
como SOLO JSON con este modelo:
{
  "resumen": string,
  "hechos_extraidos": <mismo JSON de hechos>,
  "evaluacion": <mismo JSON de evaluación>,
  "fallo_simulado": string,
  "disclaimer": string
}

Hechos (JSON):
$facts_json

Evaluación (JSON):
$eval_json

Notas:
- "fallo_simulado" debe sonar a parte dispositiva (p. ej. "SE DECLARA..."),
  pero deja claro que es una redacción basada en el texto.
- "disclaimer" debe indicar que es orientativo, no sustituye consejo jurídico y
  depende de jurisdicción.
"""
        msg = [
            SystemMessage(content=JUDGE_PROMPT),
            HumanMessage(
                content=self._tpl(
                    judge_output_format_prompt,
                    facts_json=state.get("facts_json"),
                    eval_json=state.get("eval_json"),
                ),
            ),
        ]

        # Call the LLM
        resp = self._llm.call(msg)

        # Get the response
        out_raw = resp.content

        data = self._robust_json_load(out_raw)

        out = JudgeOutput.model_validate(data)

        return {
            "output_json": out.model_dump_json(ensure_ascii=False, indent=2),
            "messages": [resp],
        }
