from atlas_ai import config
from atlas_ai.errors import AtlasError
from atlas_ai.llm.client import OpenAIClient
from atlas_ai.services.assistant import AssistantService
from atlas_ai.logging_config import configure_logging


def main() -> None:
    configure_logging()
    client = OpenAIClient(
        api_key=config.OPENAI_API_KEY,
        model=config.OPENAI_MODEL
    )
    assistant = AssistantService(llm_client=client)
    print("Atlas AI")
    print("Type 'exit' to quit.\n")

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


if __name__ == "__main__":
    main()
