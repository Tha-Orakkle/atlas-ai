from openai import OpenAI


class OpenAIClient:
    """
    Client communicate with the OpenAI API using the
    OpenAI SDK.
    """

    def __init__(self, api_key: str, model: str):
        """
        Initializes an openai client
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate(
        self,
        context: list[dict],
        tools: list[dict] | None = None
    ):
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
