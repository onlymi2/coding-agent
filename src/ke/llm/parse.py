import json
from collections.abc import Mapping
from typing import Any

from ke.llm.types import LLMResponse, Message, ToolCall


def _get(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _parse_arguments(raw: Any) -> tuple[dict[str, Any], str | None]:
    if isinstance(raw, Mapping):
        return dict(raw), None
    if raw is None or raw == "":
        return {}, None
    if not isinstance(raw, str):
        return {}, f"tool arguments 必须是 JSON 字符串或对象，实际为 {type(raw).__name__}"

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        return {}, f"tool arguments 不是合法 JSON：{exc.msg}"

    if not isinstance(parsed, dict):
        return {}, "tool arguments 的 JSON 顶层必须是对象"
    return parsed, None


def _parse_tool_call(raw_call: Any) -> ToolCall:
    function = _get(raw_call, "function", {})
    arguments, arguments_error = _parse_arguments(
        _get(function, "arguments", None)
    )
    return ToolCall(
        id=str(_get(raw_call, "id", "")),
        name=str(_get(function, "name", "")),
        arguments=arguments,
        arguments_error=arguments_error,
    )


def _optional_int(value: Any) -> int | None:
    return value if isinstance(value, int) else None


def parse_response(response: Any) -> LLMResponse:
    """Convert a Chat Completions-like response into internal types.

    Both dictionaries and SDK objects with matching attributes are accepted.
    Invalid tool argument JSON is recorded on ``ToolCall.arguments_error``.
    """

    if isinstance(response, LLMResponse):
        return response

    choices = _get(response, "choices", None)
    if not choices:
        raise ValueError("模型响应中没有 choices")

    choice = choices[0]
    raw_message = _get(choice, "message", {})
    raw_content = _get(raw_message, "content", None)
    content = raw_content if isinstance(raw_content, str) else None
    raw_tool_calls = _get(raw_message, "tool_calls", None) or []
    tool_calls = [_parse_tool_call(call) for call in raw_tool_calls]

    usage = _get(response, "usage", {}) or {}
    return LLMResponse(
        message=Message(
            role="assistant",
            content=content,
            tool_calls=tool_calls,
        ),
        finish_reason=str(_get(choice, "finish_reason", "") or ""),
        usage_prompt_tokens=_optional_int(_get(usage, "prompt_tokens", None)),
        usage_completion_tokens=_optional_int(
            _get(usage, "completion_tokens", None)
        ),
    )
