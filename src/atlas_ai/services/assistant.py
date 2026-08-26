import json
from atlas_ai.tools.registry import TOOLS
from atlas_ai.prompts import PROMPTS


class AssistantService:
    def __init__(self, llm_client):
        self.client = llm_client
        self.tools = [tool["schema"] for tool in TOOLS.values()]
        self.context = []
        self.add_to_context(
            role="developer",
            content=PROMPTS["main"]
        )

    def add_to_context(self, role: str, content: str) -> None:
        """
        Adds input/response from user/assistant to the conversation.
        Args:
            - role (str): user or assistant.
            - content (str): the actual input by user or response from
              the assistant
        """
        self.context.append({
            "role": role,
            "content": content
        })

    def execute_tools(self, response_output: list) -> list[dict]:
        """"
        Gets and executes the tool called by model.
        Args:
            - response_output: list of responses from the AI model.
        Returns:
            - list of all function_call_outputs
        """
        tools_output = []

        for item in response_output:
            if item.type != "function_call":
                continue
            tool = TOOLS.get(item.name)
            if not tool:
                tools_output.append({
                    "type": "function_call_output",
                    "call_id": item.call_id,
                    "output": json.dumps({
                        "error", f"Unknown tool: {item.name}"
                    })
                })
                continue

            args = json.loads(item.arguments)
            result = tool["function"](**args)
            tools_output.append({
                "type": "function_call_output",
                "call_id": item.call_id,
                "output": json.dumps(result)
            })

        return tools_output

    def generate_response(self, user_input: str) -> str:
        """
        Get responses from AI model. Execute tools if
        model makes tool calls.
        Args:
        - user_input: input by user
        """
        self.add_to_context("user", user_input)

        input_list = self.context.copy()

        while True:
            response = self.client.generate(
                context=input_list,
                tools=self.tools
            )
            input_list += response.output
            tools_output = self.execute_tools(response.output)
            if not tools_output:
                break
            input_list += tools_output

        self.add_to_context("assistant", response.output_text)
        return response.output_text
