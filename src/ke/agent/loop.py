import json
from dataclasses import dataclass, field
from typing import Iterator

from ke.agent.context import AgentContext
from ke.agent.events import AgentEvent
from ke.llm.protocol import LlmClient
from ke.llm.types import Message, ToolCall
from ke.safety.confirm import ConfirmationGate
from ke.tools.registry import ToolRegistry
from ke.tools.types import ToolResult


@dataclass
class AgentState:
    """Mutable state for one agent run, including external cancellation."""

    context: AgentContext = field(default_factory=AgentContext)
    max_turns: int = 30
    cancelled: bool = False
    turn: int = 0

    def __post_init__(self) -> None:
        if self.max_turns < 1:
            raise ValueError("max_turns 必须大于等于 1")

    def abort(self) -> None:
        self.cancelled = True

    @property
    def messages(self) -> list[Message]:
        """Compatibility view for callers that inspect the message history."""

        return self.context.messages


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
    summary_llm: LlmClient | None = None,
    confirmation: ConfirmationGate | None = None,
) -> Iterator[AgentEvent]:
    """Run the model-tool loop and expose every important step as an event."""

    state.context.append(Message(role="user", content=task))
    context_llm = llm if summary_llm is None else summary_llm
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
            response = llm.complete(state.context.messages, tools.schemas())
        except Exception as exc:
            yield AgentEvent(
                type="error",
                turn=state.turn,
                message=f"LLM 调用失败：{type(exc).__name__}: {exc}",
            )
            return

        assistant = response.message
        state.context.append(assistant)

        if state.cancelled:
            yield AgentEvent(
                type="error",
                turn=state.turn,
                message="用户已中止运行",
            )
            return

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
            if state.cancelled:
                break
            yield AgentEvent(
                type="tool_request",
                turn=state.turn,
                tool_call=call,
            )
            if state.cancelled:
                break

            if confirmation is not None and confirmation.requires_confirmation(call):
                permission = confirmation.create(call)
                yield AgentEvent(
                    type="tool_confirm",
                    turn=state.turn,
                    tool_call=call,
                    permission_id=permission.permission_id,
                    preview=permission.preview,
                    message=f"等待用户确认工具：{call.name}",
                )
                allowed = confirmation.wait(permission.permission_id)
                if allowed and not state.cancelled:
                    result = tools.execute(call)
                else:
                    result = ToolResult.error(
                        f"用户拒绝执行工具：{call.name}"
                    )
            else:
                result = tools.execute(call)
            state.context.append(
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

            if state.context.should_compact():
                tokens_before = state.context.estimate_tokens()
                collapsed_count = state.context.compact()
                if collapsed_count > 0:
                    tokens_after = state.context.estimate_tokens()
                    yield AgentEvent(
                        type="context_compact",
                        turn=state.turn,
                        message=(
                            f"上下文压缩：估算 tokens {tokens_before} -> "
                            f"{tokens_after}，折叠 {collapsed_count} 条旧工具结果"
                        ),
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

        if state.cancelled:
            yield AgentEvent(
                type="error",
                turn=state.turn,
                message="用户已中止运行",
            )
            return

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

        if (
            not state.cancelled
            and state.turn < state.max_turns
            and state.context.should_summarize()
        ):
            tokens_before = state.context.estimate_tokens()
            if state.context.summarize(context_llm):
                tokens_after = state.context.estimate_tokens()
                yield AgentEvent(
                    type="context_summary",
                    turn=state.turn,
                    message=(
                        f"LLM 上下文摘要：估算 tokens {tokens_before} -> "
                        f"{tokens_after}"
                    ),
                )
