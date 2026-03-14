## Divorce Judge (LangGraph)

Juez sencillo (no vinculante) para analizar sentencias de divorcio en PDF usando modelos de lenguaje (OpenAI u Ollama) y un flujo orquestado con LangGraph.

La idea es explorar **cómo la IA puede apoyar en dominios sensibles como el derecho**, generando resúmenes estructurados y evaluaciones de sentencias, siempre con un fuerte **disclaimer**: esto **no es asesoría legal** ni sustituye el criterio profesional humano.

### Requisitos

- Python 3.10+
- `virtualenv` (opcional pero recomendado)

Dependencias principales (ya listadas en `requirements.txt`):

- `langchain`, `langgraph`
- `langchain-openai`, `langchain-ollama`
- `pydantic`, `python-dotenv`
- `pypdf`

### Instalación

```bash
git clone https://github.com/franciscosanchezoliver/divorce_judge.git
cd divorce_judge

python -m venv .venv
source .venv/bin/activate  # En Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### Configuración de variables de entorno

Para usar modelos de OpenAI:

```bash
export OPENAI_API_KEY="tu_api_key"
```

Para usar modelos de Ollama:

- Tener `ollama` instalado y corriendo localmente.
- Tener descargado el modelo que vayas a usar (por ejemplo `qwen2.5:3b`).

Puedes usar un fichero `.env` en la raíz del proyecto (está ignorado en `.gitignore`) y definir ahí tus claves:

```env
OPENAI_API_KEY=tu_api_key
```

### Uso

El punto de entrada es `src/main.py`. Ejemplo:

```bash
python -m src.main \
    --file /ruta/a/tu/sentencia.pdf \
    --provider ollama \
    --model qwen2.5:3b
```

O con OpenAI:

```bash
python -m src.main \
    --file /ruta/a/tu/sentencia.pdf \
    --provider openai \
    --model gpt-4o-mini
```

El programa:

- Convierte el PDF a texto.
- Extrae hechos relevantes (partes, custodia, pensiones, etc.).
- Evalúa la coherencia y completitud de la sentencia.
- Genera un JSON estructurado con:
  - `resumen`
  - `hechos_extraidos`
  - `evaluacion`
  - `fallo_simulado`
  - `disclaimer`

La salida final se registra por `logging` (nivel `INFO`).

### Limitaciones y aviso legal

- El modelo puede cometer errores, alucinar datos o pasar por alto información relevante.
- El sistema **no sustituye** el análisis de un profesional del derecho ni constituye asesoría legal.
- El proyecto está pensado como **ejercicio educativo/exploratorio** sobre el uso de IA en contextos jurídicos.

