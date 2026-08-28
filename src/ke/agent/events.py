from dataclasses import dataclass
from typing import Literal

from ke.llm.types import ToolCall
from ke.tools.types import ToolResult


EventType = Literal[
    "turn_start",
    "tool_request",
    "tool_confirm",
    "tool_result",
    "context_compact",
    "context_summary",
    "final",
    "error",
]


@dataclass(frozen=True)
class AgentEvent:
    """One observable state transition emitted by the agent loop."""

    type: EventType
    turn: int
    message: str | None = None
    tool_call: ToolCall | None = None
    tool_result: ToolResult | None = None
    permission_id: str | None = None
    preview: str | None = None
