
import argparse
import logging

from pydantic import BaseModel, Field, ValidationError

from src.judge import DivorceJudge
from src.llm import LLM


logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Juez simple (LangGraph) para sentencias de divorcio."
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        # TODO: put here the url where the divorce pdfs are
        help="Ruta a la sentencia (.pdf).",
    )

    parser.add_argument(
        "--provider",
        type=str,
        default="ollama",
        choices=["ollama", "openai"],
        help="Proveedor del modelo.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Modelo. Ej: gpt-4o-mini (openai) o qwen2.5:3b (ollama).",
    )
    args = parser.parse_args()

    # Creation of the LLM depending on the parameters passed
    llm = LLM(provider=args.provider, model=args.model)

    # Divorce case
    judge = DivorceJudge(llm=llm)

    # Analyze the case and get the result of it
    result = judge.analyze_case(divorce_case_file_path=args.file)

    logger.info("Resultado del análisis:\n%s", result)


if __name__ == "__main__":
    main()
