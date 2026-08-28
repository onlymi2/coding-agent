from types import SimpleNamespace
from typing import Any

import pytest

import ke.llm.client as client_module
from ke.llm.client import (
    LlmError,
    OpenAICompatClient,
    message_to_openai,
    messages_to_openai,
)
from ke.llm.types import Message, ToolCall


class RecordingCompletions:
    def __init__(self, response: object = None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.requests: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> object:
        self.requests.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


def sdk_with(completions: RecordingCompletions) -> SimpleNamespace:
    return SimpleNamespace(chat=SimpleNamespace(completions=completions))


def test_internal_messages_convert_to_openai_format() -> None:
    converted = messages_to_openai(
        [
            Message(role="system", content="rules"),
            Message(role="user", content="fix it"),
            Message(
                role="assistant",
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        name="write_file",
                        arguments={"z": 2, "a": 1},
                    )
                ],
            ),
            Message(
                role="tool",
                content="written",
                tool_call_id="call_1",
                name="write_file",
            ),
            Message(role="assistant", content="done"),
        ]
    )

    assert converted[0] == {"role": "system", "content": "rules"}
    assert converted[1] == {"role": "user", "content": "fix it"}
    assert converted[2] == {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "write_file",
                    "arguments": '{"a":1,"z":2}',
                },
            }
        ],
    }
    assert converted[3] == {
        "role": "tool",
        "content": "written",
        "tool_call_id": "call_1",
        "name": "write_file",
    }
    assert converted[4] == {"role": "assistant", "content": "done"}


def test_client_passes_configuration_and_request_then_parses_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "returned_call",
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "arguments": '{"path":"hello.py"}',
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 20, "completion_tokens": 5},
    }
    completions = RecordingCompletions(response=response)
    constructor_arguments: dict[str, Any] = {}

    def fake_openai(**kwargs: Any) -> SimpleNamespace:
        constructor_arguments.update(kwargs)
        return sdk_with(completions)

    monkeypatch.setattr(client_module, "OpenAI", fake_openai)
    client = OpenAICompatClient(
        api_key="test-key",
        base_url="https://gateway.example/v1",
        model="test-model",
        timeout=17,
    )
    schemas = [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "read",
                "parameters": {"type": "object"},
            },
        }
    ]

    result = client.complete([Message(role="system", content="rules")], schemas)

    assert constructor_arguments == {
        "api_key": "test-key",
        "base_url": "https://gateway.example/v1",
        "timeout": 17,
    }
    assert completions.requests == [
        {
            "model": "test-model",
            "messages": [{"role": "system", "content": "rules"}],
            "tools": schemas,
            "tool_choice": "auto",
        }
    ]
    assert result.finish_reason == "tool_calls"
    assert result.usage_prompt_tokens == 20
    assert result.message.tool_calls[0].name == "read_file"
    assert result.message.tool_calls[0].arguments == {"path": "hello.py"}


def test_sdk_exception_is_wrapped_without_leaking_key() -> None:
    completions = RecordingCompletions(
        error=RuntimeError("request failed with test-key"),
    )
    client = OpenAICompatClient(
        api_key="test-key",
        base_url="https://gateway.example/v1",
        model="test-model",
        sdk_client=sdk_with(completions),
    )

    with pytest.raises(LlmError, match="RuntimeError") as captured:
        client.complete([Message(role="user", content="hello")], [])

    assert "test-key" not in str(captured.value)


def test_client_omits_tool_fields_when_tools_are_empty() -> None:
    completions = RecordingCompletions(
        response={
            "choices": [
                {
                    "message": {"content": "summary"},
                    "finish_reason": "stop",
                }
            ]
        }
    )
    client = OpenAICompatClient(
        api_key="test-key",
        base_url="https://gateway.example/v1",
        model="test-model",
        sdk_client=sdk_with(completions),
    )

    result = client.complete([Message(role="user", content="summarize")], [])

    request = completions.requests[0]
    assert "tools" not in request
    assert "tool_choice" not in request
    assert result.message.content == "summary"


def test_sdk_initialization_exception_is_wrapped_without_leaking_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken_openai(**kwargs: Any) -> None:
        raise RuntimeError(f"failed with {kwargs['api_key']}")

    monkeypatch.setattr(client_module, "OpenAI", broken_openai)

    with pytest.raises(LlmError, match="初始化失败") as captured:
        OpenAICompatClient(
            api_key="test-key",
            base_url="https://gateway.example/v1",
            model="test-model",
        )

    assert "test-key" not in str(captured.value)


def test_plain_assistant_none_content_converts_to_empty_text() -> None:
    assert message_to_openai(Message(role="assistant")) == {
        "role": "assistant",
        "content": "",
    }
