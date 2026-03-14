import logging

from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama


logger = logging.getLogger(__name__)


class LLM:

    def __init__(
        self,
        provider,
        model,
        temperature: float = 0.0
    ):
        self._llm = None

        # Select the model to use
        if provider == "openai":
            self._llm = ChatOpenAI(model=model, temperature=temperature)
        elif provider == "ollama":
            self._llm = ChatOllama(model=model, temperature=temperature)

    def call(self, instruction):
        """Execute the llm call"""
        logger.info("Calling the LLM with the following instruction: %s", instruction)

        llm_response = self._llm.invoke(instruction)

        logger.info("Response metadata: %s", llm_response.response_metadata)
        logger.info("AI response content: %s", llm_response.content)

        return llm_response
