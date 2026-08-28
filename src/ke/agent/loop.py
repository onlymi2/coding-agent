import json
from dataclasses import dataclass, field
from typing import Iterator

from ke.agent.events import AgentEvent
from ke.llm.protocol import LlmClient
from ke.llm.types import Message, ToolCall
from ke.tools.registry import ToolRegistry


@dataclass
class AgentState:
    """Mutable state for one agent run, including external cancellation."""

    messages: list[Message] = field(default_factory=list)
    max_turns: int = 30
    cancelled: bool = False
    turn: int = 0

    def __post_init__(self) -> None:
        if self.max_turns < 1:
            raise ValueError("max_turns 必须大于等于 1")

    def abort(self) -> None:
        self.cancelled = True


def _fingerprint(call: ToolCall) -> tuple[str, str]:
    arguments = json.dumps(
        call.arguments,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=repr,
    )
    return call.name, arguments


def run_agent(
    task: str,
    llm: LlmClient,
    tools: ToolRegistry,
    state: AgentState,
) -> Iterator[AgentEvent]:
    """Run the model-tool loop and expose every important step as an event."""

    state.messages.append(Message(role="user", content=task))
    consecutive_errors = 0
    last_fingerprint: tuple[str, str] | None = None
    repeat_count = 0

    while True:
        if state.cancelled:
            yield AgentEvent(
                type="error",
                turn=state.turn,
                message="用户已中止运行",
            )
            return

        if state.turn >= state.max_turns:
            yield AgentEvent(
                type="final",
                turn=state.turn,
                message=f"已达到最大轮数（{state.max_turns}），停止运行",
            )
            return

        state.turn += 1
        yield AgentEvent(type="turn_start", turn=state.turn)

        try:
            response = llm.complete(state.messages, tools.schemas())
        except Exception as exc:
            yield AgentEvent(
                type="error",
                turn=state.turn,
                message=f"LLM 调用失败：{type(exc).__name__}: {exc}",
            )
            return

        assistant = response.message
        state.messages.append(assistant)

        if not assistant.tool_calls:
            yield AgentEvent(
                type="final",
                turn=state.turn,
                message=assistant.content or "",
            )
            return

        error_limit_reached = False
        doom_loop_reached = False
        for call in assistant.tool_calls:
            yield AgentEvent(
                type="tool_request",
                turn=state.turn,
                tool_call=call,
            )

            result = tools.execute(call)
            state.messages.append(
                Message(
                    role="tool",
                    content=result.content,
                    tool_call_id=call.id,
                    name=call.name,
                )
            )
            yield AgentEvent(
                type="tool_result",
                turn=state.turn,
                tool_call=call,
                tool_result=result,
            )

            consecutive_errors = consecutive_errors + 1 if result.is_error else 0
            fingerprint = _fingerprint(call)
            repeat_count = (
                repeat_count + 1 if fingerprint == last_fingerprint else 1
            )
            last_fingerprint = fingerprint

            if consecutive_errors >= 3:
                error_limit_reached = True
            if repeat_count >= 3:
                doom_loop_reached = True

        if error_limit_reached:
            yield AgentEvent(
                type="error",
                turn=state.turn,
                message="工具连续失败 3 次，停止运行",
            )
            return
        if doom_loop_reached:
            yield AgentEvent(
                type="error",
                turn=state.turn,
                message="检测到 doom loop：同一工具和参数连续执行 3 次",
            )
            return
