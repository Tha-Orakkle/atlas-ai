from openai import OpenAI
from atlas_ai.config import OPENAI_API_KEY


class OpenAIClient:
    """
    Client communicate with the OpenAI API using the
    OpenAI SDK.
    """

    def __init__(self):
        """
        Initializes an openai client
        """
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model = "gpt-5.4-mini"

    def generate(self, context: list[dict], tools: list[dict] = None):
        """
        Communicates with the OpenAI responses API.
        Args:
            - context (list): list of conversation history,
              function calls and function call outputs
            - tools (list): list of all available tools to the
              nmodel
        """
        return self.client.responses.create(
            model=self.model,
            input=context,
            tools=tools
        )
