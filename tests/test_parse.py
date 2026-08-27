from types import SimpleNamespace as NS

from ke.llm.parse import parse_response


def test_parse_plain_assistant_text_and_usage() -> None:
    raw = {
        "choices": [
            {
                "message": {"role": "assistant", "content": "任务完成。"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 12, "completion_tokens": 4},
    }

    result = parse_response(raw)

    assert result.message.role == "assistant"
    assert result.message.content == "任务完成。"
    assert result.message.tool_calls == []
    assert result.finish_reason == "stop"
    assert result.usage_prompt_tokens == 12
    assert result.usage_completion_tokens == 4


def test_parse_multiple_tool_calls_in_order() -> None:
    raw = NS(
        choices=[
            NS(
                message=NS(
                    content=None,
                    tool_calls=[
                        NS(
                            id="call_1",
                            function=NS(
                                name="write_file",
                                arguments='{"path":"hello.py","content":"print(1)"}',
                            ),
                        ),
                        NS(
                            id="call_2",
                            function=NS(
                                name="bash",
                                arguments='{"command":"python hello.py"}',
                            ),
                        ),
                    ],
                ),
                finish_reason="tool_calls",
            )
        ],
        usage=None,
    )

    result = parse_response(raw)

    assert result.message.content is None
    assert [call.name for call in result.message.tool_calls] == [
        "write_file",
        "bash",
    ]
    assert result.message.tool_calls[0].arguments["path"] == "hello.py"
    assert result.message.tool_calls[1].arguments == {
        "command": "python hello.py"
    }
    assert all(call.arguments_error is None for call in result.message.tool_calls)


def test_invalid_tool_arguments_are_recorded_without_crashing() -> None:
    raw = {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "bad_call",
                            "function": {
                                "name": "read_file",
                                "arguments": "{not-json",
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ]
    }

    result = parse_response(raw)
    call = result.message.tool_calls[0]

    assert call.arguments == {}
    assert call.arguments_error is not None
    assert "不是合法 JSON" in call.arguments_error


def test_empty_content_without_tools_is_valid() -> None:
    raw = {
        "choices": [
            {
                "message": {"content": None, "tool_calls": []},
                "finish_reason": "stop",
            }
        ]
    }

    result = parse_response(raw)

    assert result.message.content is None
    assert result.message.tool_calls == []
