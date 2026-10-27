from decimal import Decimal
from unittest.mock import Mock

import atlas_ai.main as main_module
from atlas_ai.errors import AtlasError
from atlas_ai.llm.streaming import (
    ResponseCompleted,
    StreamError,
    TextDelta,
)
from atlas_ai.models import AssistantMessage, TokenUsage

# def configure_main(monkeypatch, events):
#     assistant = Mock()
#     assistant.astream_response.return_value = iter(events)

#     monkeypatch.setattr(
#         main_module,
#         "configure_logging",
#         Mock(),
#     )
#     monkeypatch.setattr(
#         main_module,
#         "OpenAIClient",
#         Mock(),
#     )
#     monkeypatch.setattr(
#         main_module,
#         "ToolExecutor",
#         Mock(),
#     )
#     monkeypatch.setattr(
#         main_module,
#         "AssistantService",
#         Mock(return_value=assistant),
#     )

#     monkeypatch.setattr(main_module.config, "OPENAI_API_KEY", "test-key")
#     monkeypatch.setattr(main_module.config, "OPENAI_MODEL", "test-model")

#     return assistant


def configure_main(monkeypatch, events):
    """
    Configuration for asynchronous execution and
    streaming communication
    """
    assistant = Mock()

    async def fake_astream_response(user_input):
        for event in events:
            yield event

    assistant.astream_response = Mock(
        side_effect=fake_astream_response
    )

    monkeypatch.setattr(
        main_module,
        "configure_logging",
        Mock(),
    )
    monkeypatch.setattr(main_module, "OpenAIClient", Mock())
    monkeypatch.setattr(main_module, "ToolExecutor", Mock())
    monkeypatch.setattr(
        main_module,
        "AssistantService",
        Mock(return_value=assistant),
    )

    monkeypatch.setattr(
        main_module.config,
        "OPENAI_API_KEY",
        "test-key",
    )
    monkeypatch.setattr(
        main_module.config,
        "OPENAI_MODEL",
        "test-model",
    )

    return assistant


def test_main_prints_streamed_text(
    monkeypatch,
    capsys,
):
    final_message = AssistantMessage(text="Hello world.")
    assistant = configure_main(
        monkeypatch,
        [
            TextDelta(text="Hello "),
            TextDelta(text="world."),
            ResponseCompleted(message=final_message),
        ],
    )

    inputs = iter(["Say hello.", "exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    main_module.main()

    output = capsys.readouterr().out

    assert "Atlas: Hello world." in output
    assert "Goodbye." in output
    assistant.astream_response.assert_called_once_with("Say hello.")


def test_main_displays_stream_error(
    monkeypatch,
    capsys,
):
    assistant = configure_main(
        monkeypatch,
        [
            TextDelta(text="Partial answer"),
            StreamError(error=AtlasError("Provider unavailable")),
        ],
    )

    inputs = iter(["Hello.", "exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    main_module.main()

    output = capsys.readouterr().out

    assert "Partial answer" in output
    assert "Atlas encountered an error: Provider unavailable" in output
    assistant.astream_response.assert_called_once_with("Hello.")


def test_main_handles_atlas_error_raised_during_streaming(
    monkeypatch,
    capsys,
):
    assistant = configure_main(monkeypatch, [])
    assistant.astream_response.side_effect = AtlasError("Unexpected failure")

    inputs = iter(["Hello.", "exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    main_module.main()

    output = capsys.readouterr().out

    assert "Atlas encountered an error: Unexpected failure" in output


def test_main_exits_without_starting_a_stream(
    monkeypatch,
    capsys,
):
    assistant = configure_main(monkeypatch, [])
    monkeypatch.setattr("builtins.input", lambda _: "exit")

    main_module.main()

    assert "Goodbye." in capsys.readouterr().out
    assistant.astream_response.assert_not_called()


def test_build_cost_calculator_registers_provider_prefixed_model(
    monkeypatch,
):
    monkeypatch.setattr(
        main_module.config, "OPENAI_MODEL", "test-model"
    )
    monkeypatch.setattr(
        main_module.config, "OPENAI_INPUT_PRICE_PER_MILLION", "2.00"
    )
    monkeypatch.setattr(
        main_module.config, "OPENAI_OUTPUT_PRICE_PER_MILLION", "8.00"
    )
    monkeypatch.setattr(
        main_module.config,
        "OPENAI_CACHED_INPUT_PRICE_PER_MILLION",
        "0.50",
    )

    calculator = main_module.build_cost_calculator()

    usage = TokenUsage(1000, 300, 1300)

    assert calculator.estimate("openai:test-model", usage) == (
        Decimal("0.0044")
    )
    assert calculator.estimate("test-model", usage) is None


def test_build_cost_calculator_without_prices_returns_unknown_cost(
    monkeypatch,
):
    monkeypatch.setattr(
        main_module.config, "OPENAI_MODEL", "test-model"
    )
    monkeypatch.setattr(
        main_module.config, "OPENAI_INPUT_PRICE_PER_MILLION", None
    )
    monkeypatch.setattr(
        main_module.config, "OPENAI_OUTPUT_PRICE_PER_MILLION", None
    )
    monkeypatch.setattr(
        main_module.config,
        "OPENAI_CACHED_INPUT_PRICE_PER_MILLION",
        None,
    )

    calculator = main_module.build_cost_calculator()

    assert calculator.estimate(
        "openai:test-model",
        TokenUsage(1000, 300, 1300),
    ) is None
