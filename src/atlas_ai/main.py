from atlas_ai.client import get_response
from atlas_ai.prompts import PROMPTS


def main() -> None:
    context = [
        {
            "role": "developer",
            "content": PROMPTS["main"]
        }
    ]
    print("Atlas AI")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ")

        if user_input.strip().lower() == "exit":
            print("Goodbye.")
            break

        context.append({
            "role": "user",
            "content": user_input
        })

        output = get_response(context)

        print(f"\nAtlas: {output}\n")

        context.append({
            "role": "assistant",
            "content": output
        })


if __name__ == "__main__":
    main()
