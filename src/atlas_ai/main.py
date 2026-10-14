from atlas_ai import config
from atlas_ai.errors import AtlasError
from atlas_ai.llm.open_ai import OpenAIClient
from atlas_ai.llm.streaming import StreamError, TextDelta
from atlas_ai.logging_config import configure_logging
from atlas_ai.services.assistant import AssistantService
from atlas_ai.tools.executor import ToolExecutor
from atlas_ai.tools.registry import TOOLS


def build_assistant() -> AssistantService:
    client = OpenAIClient(
        api_key=config.OPENAI_API_KEY,
        model=config.OPENAI_MODEL,
        tool_schema=[
            tool["schema"]
            for tool in TOOLS.values()
        ]
    )

    tool_executor = ToolExecutor(tools_registry=TOOLS)

    return AssistantService(
        llm_client=client,
        tool_executor=tool_executor
    )


def run_atlas(assistant: AssistantService):
    while True:
        user_input = input("You: ")

        if user_input.strip().lower() == "exit":
            print("Goodbye.")
            break

        try:
            response = assistant.generate_response(user_input)
        except AtlasError as exc:
            response = f"Atlas encountered an error: {exc}"

        print(f"\nAtlas: {response}\n")


def run_atlas_streaming(assistant: AssistantService):
    while True:
        user_input = input("You: ")

        if user_input.strip().lower() == "exit":
            print("Goodbye.")
            break

        print("\nAtlas: ", end="", flush=True)
        stream_error = None

        try:

            for event in assistant.stream_response(user_input):
                if isinstance(event, TextDelta):
                    print(event.text, end="", flush=True)

                elif isinstance(event, StreamError):
                    stream_error = event.error
                    break

        except AtlasError as exc:
            stream_error = exc

        if stream_error is not None:
            print(
                f"\nAtlas encountered an error: {stream_error}\n"
            )
        else:
            print("\n")


def main() -> None:
    configure_logging()
    assistant = build_assistant()

    print("Atlas AI")
    print("Type 'exit' to quit.\n")

    # run_atlas(assistant=assistant)
    run_atlas_streaming(assistant=assistant)


if __name__ == "__main__":
    main()
