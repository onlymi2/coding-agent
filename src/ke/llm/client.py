import json
from collections.abc import Sequence
from typing import Any

from openai import OpenAI

from ke.llm.parse import parse_response
from ke.llm.protocol import ToolSchema
from ke.llm.types import LLMResponse, Message, ToolCall


DEFAULT_LLM_TIMEOUT_SECONDS = 60.0


class LlmError(RuntimeError):
    """A provider-independent failure raised by an LLM client."""


def _tool_call_to_openai(call: ToolCall) -> dict[str, Any]:
    return {
        "id": call.id,
        "type": "function",
        "function": {
            "name": call.name,
            "arguments": json.dumps(
                call.arguments,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=repr,
            ),
        },
    }


def message_to_openai(message: Message) -> dict[str, Any]:
    """Convert one provider-independent Message to Chat Completions format."""

    if message.role in {"system", "user"}:
        return {"role": message.role, "content": message.content or ""}

    if message.role == "assistant":
        converted: dict[str, Any] = {
            "role": "assistant",
            "content": message.content,
        }
        if message.tool_calls:
            converted["tool_calls"] = [
                _tool_call_to_openai(call) for call in message.tool_calls
            ]
        elif converted["content"] is None:
            converted["content"] = ""
        return converted

    converted = {
        "role": "tool",
        "content": message.content or "",
        "tool_call_id": message.tool_call_id or "",
    }
    if message.name:
        converted["name"] = message.name
    return converted


def messages_to_openai(messages: Sequence[Message]) -> list[dict[str, Any]]:
    return [message_to_openai(message) for message in messages]


class OpenAICompatClient:
    """Chat Completions client configured only by endpoint, model, and key."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        timeout: float = DEFAULT_LLM_TIMEOUT_SECONDS,
        sdk_client: Any | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("api_key 不能为空")
        if not base_url:
            raise ValueError("base_url 不能为空")
        if not model:
            raise ValueError("model 不能为空")
        if timeout <= 0:
            raise ValueError("timeout 必须大于 0")

        self.base_url = base_url
        self.model = model
        self.timeout = timeout
        if sdk_client is not None:
            self._client = sdk_client
        else:
            try:
                self._client = OpenAI(
                    api_key=api_key,
                    base_url=base_url,
                    timeout=timeout,
                )
            except Exception as exc:
                raise LlmError(
                    f"LLM 客户端初始化失败（{type(exc).__name__}）"
                ) from None

    def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSchema],
    ) -> LLMResponse:
        try:
            request: dict[str, Any] = {
                "model": self.model,
                "messages": messages_to_openai(messages),
            }
            if tools:
                request["tools"] = list(tools)
                request["tool_choice"] = "auto"
            response = self._client.chat.completions.create(**request)
            return parse_response(response)
        except Exception as exc:
            raise LlmError(
                f"LLM 请求失败（{type(exc).__name__}）"
            ) from None
