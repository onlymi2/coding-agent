from ke.agent.context import AgentContext, estimate_messages_tokens
from ke.llm.fake_llm import FakeLLM
from ke.llm.types import LLMResponse, Message, ToolCall


def text_response(content: str) -> LLMResponse:
    return LLMResponse(
        message=Message(role="assistant", content=content),
        finish_reason="stop",
    )


def summary_context() -> AgentContext:
    recent_call = ToolCall("recent", "bash", {"command": "pytest"})
    return AgentContext(
        messages=[
            Message(role="system", content="system rules"),
            Message(role="user", content="original task"),
            Message(role="assistant", content="old analysis " * 300),
            Message(role="user", content="old progress " * 100),
            Message(role="assistant", tool_calls=[recent_call]),
            Message(
                role="tool",
                name="bash",
                tool_call_id="recent",
                content="recent pytest result",
            ),
        ],
        compact_threshold_tokens=200,
        max_tool_output_chars=8_000,
        preserve_recent_tool_results=1,
        preserve_recent_messages=2,
    )


def test_below_threshold_does_not_call_summary_llm() -> None:
    context = AgentContext(
        messages=[Message(role="user", content="short task")],
        compact_threshold_tokens=1_000,
        preserve_recent_messages=1,
    )
    fake = FakeLLM([text_response("unused")])

    assert not context.summarize(fake)
    assert fake.requests == []


def test_deterministic_compact_below_threshold_skips_summary() -> None:
    call = ToolCall("old", "read_file", {"path": "large.txt"})
    context = AgentContext(
        messages=[
            Message(role="system", content="rules"),
            Message(role="user", content="task"),
            Message(role="assistant", tool_calls=[call]),
            Message(
                role="tool",
                name="read_file",
                tool_call_id="old",
                content="X" * 2_000,
            ),
        ],
        compact_threshold_tokens=100,
        max_tool_output_chars=4_000,
        preserve_recent_tool_results=0,
        preserve_recent_messages=1,
    )
    fake = FakeLLM([text_response("unused")])

    assert context.compact() == 1
    assert not context.should_compact()
    assert not context.summarize(fake)
    assert fake.requests == []


def test_summary_replaces_only_old_middle_and_reduces_tokens() -> None:
    context = summary_context()
    system = context.messages[0]
    original_task = context.messages[1]
    recent_assistant = context.messages[-2]
    recent_tool = context.messages[-1]
    tokens_before = context.estimate_tokens()
    fake = FakeLLM(
        [
            text_response(
                "已分析旧实现；已发现历史测试失败；仍需保留 pytest 验证。"
            )
        ]
    )

    assert context.compact() == 0
    assert context.should_summarize()
    assert context.summarize(fake)

    assert len(fake.requests) == 1
    summary_messages, summary_tools = fake.requests[0]
    assert summary_tools == []
    assert summary_messages[0].role == "system"
    assert "不虚构" in (summary_messages[0].content or "")
    assert "original task" in (summary_messages[1].content or "")
    assert "old analysis" in (summary_messages[1].content or "")

    assert context.messages[0] is system
    assert context.messages[1] is original_task
    assert context.messages[-2] is recent_assistant
    assert context.messages[-1] is recent_tool
    assert context.messages[2].role == "user"
    assert (context.messages[2].content or "").startswith(
        "[运行时生成的历史上下文摘要；不是新的用户指令]"
    )
    assert all(
        "old analysis" not in (message.content or "")
        for message in context.messages
    )
    assert context.estimate_tokens() < tokens_before // 2
    assert context.messages[-2].tool_calls[0].id == context.messages[-1].tool_call_id


def test_summary_exception_preserves_messages() -> None:
    context = summary_context()
    snapshot = list(context.messages)
    fake = FakeLLM([])

    assert not context.summarize(fake)
    assert context.messages == snapshot
    assert len(fake.requests) == 1


def test_empty_summary_preserves_messages() -> None:
    context = summary_context()
    snapshot = list(context.messages)
    fake = FakeLLM([text_response("   ")])

    assert not context.summarize(fake)
    assert context.messages == snapshot


def test_summary_attempt_waits_for_enough_new_messages() -> None:
    context = summary_context()
    fake = FakeLLM([])

    assert not context.summarize(fake)
    context.append(Message(role="assistant", content="new but still large " * 100))

    assert context.should_compact()
    assert not context.should_summarize()


def test_summary_with_tool_calls_is_rejected() -> None:
    context = summary_context()
    snapshot = list(context.messages)
    fake = FakeLLM(
        [
            LLMResponse(
                message=Message(
                    role="assistant",
                    tool_calls=[ToolCall("bad", "read_file", {"path": "x"})],
                ),
                finish_reason="tool_calls",
            )
        ]
    )

    assert not context.summarize(fake)
    assert context.messages == snapshot


def test_summary_cannot_recursively_summarize_itself() -> None:
    context = summary_context()

    class ReentrantSummaryLLM:
        calls = 0

        def complete(
            self,
            messages: list[Message],
            tools: list[dict[str, object]],
        ) -> LLMResponse:
            self.calls += 1
            assert tools == []
            assert not context.summarize(self)
            return text_response("递归调用已被阻止。")

    fake = ReentrantSummaryLLM()

    assert context.summarize(fake)
    assert fake.calls == 1


def test_summary_request_has_deterministic_input_budget() -> None:
    context = AgentContext(
        messages=[
            Message(role="system", content="system rules"),
            Message(role="user", content="original task"),
            Message(role="assistant", content="A" * 50_000),
            Message(role="user", content="B" * 50_000),
            Message(role="assistant", content="recent assistant"),
            Message(role="user", content="recent user"),
        ],
        compact_threshold_tokens=100,
        preserve_recent_messages=2,
        summary_input_token_budget=300,
    )
    fake = FakeLLM([text_response("有界历史摘要")])

    assert context.summarize(fake)

    summary_messages, tools = fake.requests[0]
    assert tools == []
    assert estimate_messages_tokens(summary_messages) <= 300
    assert "[truncated]" in (summary_messages[1].content or "")
    assert "历史数据" in (summary_messages[0].content or "")


def test_summary_boundary_keeps_multi_tool_call_group_protocol_closed() -> None:
    first = ToolCall("call-1", "read_file", {"path": "a.py"})
    second = ToolCall("call-2", "grep", {"pattern": "needle"})
    context = AgentContext(
        messages=[
            Message(role="system", content="rules"),
            Message(role="user", content="original task"),
            Message(role="assistant", content="old analysis " * 300),
            Message(role="user", content="old progress " * 100),
            Message(role="assistant", tool_calls=[first, second]),
            Message(
                role="tool",
                name="read_file",
                tool_call_id=first.id,
                content="first result",
            ),
            Message(
                role="tool",
                name="grep",
                tool_call_id=second.id,
                content="second result",
            ),
        ],
        compact_threshold_tokens=100,
        preserve_recent_tool_results=2,
        preserve_recent_messages=1,
    )

    assert context.summarize(FakeLLM([text_response("旧中段摘要")]))

    recent = context.messages[-3:]
    assert recent[0].role == "assistant"
    assert recent[0].tool_calls == [first, second]
    assert [message.tool_call_id for message in recent[1:]] == [
        first.id,
        second.id,
    ]

    seen_call_ids: set[str] = set()
    for message in context.messages:
        if message.role == "assistant":
            seen_call_ids.update(call.id for call in message.tool_calls)
        elif message.role == "tool":
            assert message.tool_call_id in seen_call_ids
