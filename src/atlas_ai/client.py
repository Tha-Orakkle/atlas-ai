from openai import OpenAI
from atlas_ai.config import OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)


def get_response(context: dict[str, str]) -> str:
    response = client.responses.create(
        model="gpt-5.4-mini",
        input=context
    )

    return response.output_text
