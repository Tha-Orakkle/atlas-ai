import json
from openai import OpenAI
from atlas_ai.config import OPENAI_API_KEY
from atlas_ai.tools.registry import TOOLS

client = OpenAI(api_key=OPENAI_API_KEY)


def execute_tools(response_output: list) -> list[dict]:
    """
    Gets and executes the tool called by model.
    Args:
        - response_output: list of responses from the AI model.
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


def get_response(context: list[dict]) -> str:
    """
    Get responses from AI model. Execute tools if
    model makes tool calls.
    Args:
        - context: conversation context.
    """

    input_list = context.copy()

    while True:
        response = client.responses.create(
            model="gpt-5.4-mini",
            input=input_list,
            tools=[tool["schema"] for tool in TOOLS.values()]
        )
        input_list += response.output

        tools_output = execute_tools(response.output)
        if not tools_output:
            break
        input_list += tools_output

    return response.output_text
