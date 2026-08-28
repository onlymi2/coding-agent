from ke.agent.context import (
    MESSAGE_OVERHEAD_TOKENS,
    AgentContext,
    estimate_message_tokens,
)
from ke.llm.types import Message, ToolCall


def test_should_compact_is_false_below_threshold() -> None:
    context = AgentContext(compact_threshold_tokens=100)
    context.append(Message(role="user", content="short task"))

    assert not context.should_compact()


def test_should_compact_is_true_above_threshold() -> None:
    context = AgentContext(compact_threshold_tokens=20)
    context.append(Message(role="user", content="x" * 100))

    assert context.should_compact()


def test_token_estimate_includes_tool_name_and_arguments() -> None:
    plain = Message(role="assistant")
    with_tool = Message(
        role="assistant",
        tool_calls=[
            ToolCall(
                id="call_1",
                name="read_file",
                arguments={"path": "a/very/long/path/to/a/file.txt"},
            )
        ],
    )

    assert estimate_message_tokens(plain) == MESSAGE_OVERHEAD_TOKENS
    assert estimate_message_tokens(with_tool) > estimate_message_tokens(plain)


def test_long_tool_message_is_truncated_with_shared_utility() -> None:
    context = AgentContext(
        compact_threshold_tokens=10_000,
        max_tool_output_chars=100,
    )

    context.append(
        Message(role="tool", name="read_file", content="A" * 500 + "Z" * 500)
    )

    content = context.messages[0].content or ""
    assert len(content) == 100
    assert "[truncated]" in content
    assert content.startswith("A")
    assert content.endswith("Z")


def test_compact_folds_old_tools_and_preserves_important_context() -> None:
    system = Message(role="system", content="system rules")
    original_task = Message(role="user", content="original task")
    old_call = ToolCall("old_call", "read_file", {"path": "old.txt"})
    recent_call = ToolCall("recent_call", "bash", {"command": "pytest"})
    context = AgentContext(
        messages=[
            system,
            original_task,
            Message(role="assistant", tool_calls=[old_call]),
            Message(
                role="tool",
                name="read_file",
                tool_call_id="old_call",
                content="old observation " * 200,
            ),
            Message(role="assistant", tool_calls=[recent_call]),
            Message(
                role="tool",
                name="bash",
                tool_call_id="recent_call",
                content="recent observation " * 10,
            ),
        ],
        compact_threshold_tokens=300,
        max_tool_output_chars=4_000,
        preserve_recent_tool_results=1,
    )
    tokens_before = context.estimate_tokens()

    collapsed_count = context.compact()
    tokens_after = context.estimate_tokens()

    assert collapsed_count == 1
    assert context.messages[0] is system
    assert context.messages[0].role == "system"
    assert context.messages[1] is original_task
    assert context.messages[2].tool_calls == [old_call]
    assert context.messages[3].collapsed
    assert context.messages[3].content == "[旧工具结果已折叠：read_file]"
    assert len(context.messages[3].content or "") < len("old observation " * 200)
    assert not context.messages[5].collapsed
    assert context.messages[5].content == "recent observation " * 10
    assert tokens_after < tokens_before // 2

    snapshot = [
        (message.role, message.content, message.collapsed)
        for message in context.messages
    ]
    assert context.compact() == 0
    assert snapshot == [
        (message.role, message.content, message.collapsed)
        for message in context.messages
    ]


def test_system_message_cannot_be_appended_after_conversation() -> None:
    context = AgentContext(messages=[Message(role="user", content="task")])

    try:
        context.append(Message(role="system", content="late system"))
    except ValueError as exc:
        assert "最前面" in str(exc)
    else:
        raise AssertionError("late system message should be rejected")


def test_compact_preserves_multi_tool_call_protocol_nodes() -> None:
    first = ToolCall("call-1", "read_file", {"path": "a.py"})
    second = ToolCall("call-2", "grep", {"pattern": "needle"})
    context = AgentContext(
        messages=[
            Message(role="user", content="inspect"),
            Message(role="assistant", tool_calls=[first, second]),
            Message(
                role="tool",
                name="read_file",
                tool_call_id=first.id,
                content="A" * 2_000,
            ),
            Message(
                role="tool",
                name="grep",
                tool_call_id=second.id,
                content="B" * 2_000,
            ),
        ],
        compact_threshold_tokens=50,
        preserve_recent_tool_results=0,
        max_tool_output_chars=4_000,
    )

    assert context.compact() == 2
    assert context.messages[1].tool_calls == [first, second]
    tool_messages = [message for message in context.messages if message.role == "tool"]
    assert [message.tool_call_id for message in tool_messages] == [
        first.id,
        second.id,
    ]
    assert all(message.collapsed for message in tool_messages)

    seen_call_ids: set[str] = set()
    for message in context.messages:
        if message.role == "assistant":
            seen_call_ids.update(call.id for call in message.tool_calls)
        elif message.role == "tool":
            assert message.tool_call_id in seen_call_ids
