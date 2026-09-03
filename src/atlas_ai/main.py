import logging
from atlas_ai import config
from atlas_ai.llm.client import OpenAIClient
from atlas_ai.services.assistant import AssistantService

logging.basicConfig(level=logging.INFO)


def main() -> None:
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

        response = assistant.generate_response(user_input)
        print(f"\nAtlas: {response}\n")


if __name__ == "__main__":
    main()
