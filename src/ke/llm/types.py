from dataclasses import dataclass, field
from typing import Any, Literal


Role = Literal["system", "user", "assistant", "tool"]


@dataclass
class ToolCall:
    """A model request to invoke one named local tool."""

    id: str
    name: str
    arguments: dict[str, Any]
    arguments_error: str | None = None


@dataclass
class Message:
    """A provider-independent conversation message."""

    role: Role
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None
    collapsed: bool = False


@dataclass
class LLMResponse:
    """The normalized result of one LLM completion."""

    message: Message
    finish_reason: str
    usage_prompt_tokens: int | None = None
    usage_completion_tokens: int | None = None
