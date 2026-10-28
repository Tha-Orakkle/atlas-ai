import asyncio
from decimal import Decimal

from atlas_ai import config
from atlas_ai.cost import CostCalculator, ModelPricing
from atlas_ai.errors import AtlasError
from atlas_ai.llm.open_ai import OpenAIClient
from atlas_ai.llm.streaming import StreamError, TextDelta
from atlas_ai.logging_config import configure_logging
from atlas_ai.services.assistant import AssistantService
from atlas_ai.tools.executor import ToolExecutor
from atlas_ai.tools.registry import TOOLS


def build_cost_calculator() -> CostCalculator:
    pricing = {}

    input_price = config.OPENAI_INPUT_PRICE_PER_MILLION
    output_price = config.OPENAI_OUTPUT_PRICE_PER_MILLION
    cached_input_price = config.OPENAI_CACHED_INPUT_PRICE_PER_MILLION

    if input_price is not None and output_price is not None:
        pricing[f"openai:{config.OPENAI_MODEL}"] = ModelPricing(
            input_per_million=Decimal(input_price),
            output_per_million=Decimal(output_price),
            cached_input_per_million=(
                Decimal(cached_input_price)
                if cached_input_price is not None
                else None
            )
        )

    return CostCalculator(pricing)


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
        tool_executor=tool_executor,
        provider="openai",
        model=config.OPENAI_MODEL,
        cost_calculator=build_cost_calculator()
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
                f"Atlas encountered an error: {stream_error}\n"
            )
        else:
            print("\n")


async def run_atlas_astreaming(assistant: AssistantService):
    while True:
        user_input = input("You: ")

        if user_input.strip().lower() == "exit":
            print("Goodbye.")
            break

        print("\nAtlas: ", end="", flush=True)
        stream_error = None

        try:
            async for event in assistant.astream_response(user_input):
                if isinstance(event, TextDelta):
                    print(event.text, end="", flush=True)

                elif isinstance(event, StreamError):
                    stream_error = event.error
                    break

        except AtlasError as exc:
            stream_error = exc

        if stream_error is not None:
            print(
                f"Atlas encountered an error: {stream_error}\n"
            )
        else:
            print("\n")


def main() -> None:
    configure_logging()
    assistant = build_assistant()

    print("Atlas AI")
    print("Type 'exit' to quit.\n")

    # run atlas synchronously without streaming LLM response
    # run_atlas(assistant=assistant)

    # run atlas synchronously while streaming LLM response
    # run_atlas_streaming(assistant=assistant)

    # run atlas asynchronously while streaming LLM response
    asyncio.run(run_atlas_astreaming(assistant=assistant))


if __name__ == "__main__":
    main()
