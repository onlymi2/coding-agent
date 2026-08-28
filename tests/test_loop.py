from pathlib import Path
import sys

import pytest

from ke.agent.context import AgentContext
from ke.agent.loop import (
    VERIFICATION_REMINDER,
    AgentState,
    _is_verification_command,
    run_agent,
)
from ke.llm.fake_llm import FakeLLM
from ke.llm.types import LLMResponse, Message, ToolCall
from ke.safety.sandbox import WorkspaceSandbox
from ke.tools.registry import ToolRegistry


PYTEST_COMMAND = f'"{sys.executable}" -m pytest -q'


def tool_response(*calls: ToolCall) -> LLMResponse:
    return LLMResponse(
        message=Message(role="assistant", tool_calls=list(calls)),
        finish_reason="tool_calls",
    )


def text_response(content: str) -> LLMResponse:
    return LLMResponse(
        message=Message(role="assistant", content=content),
        finish_reason="stop",
    )


def call(call_id: str, name: str, **arguments: object) -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments=arguments)


@pytest.fixture
def registry(tmp_path: Path) -> ToolRegistry:
    return ToolRegistry(WorkspaceSandbox(tmp_path))


def test_agent_runs_tools_across_turns_then_finishes(
    registry: ToolRegistry,
    tmp_path: Path,
) -> None:
    fake = FakeLLM(
        [
            tool_response(
                call("write", "write_file", path="hello.txt", content="hello")
            ),
            tool_response(call("read", "read_file", path="hello.txt")),
            tool_response(
                call("verify", "bash", command="python -m compileall .")
            ),
            text_response("任务完成"),
        ]
    )
    state = AgentState()

    events = list(run_agent("创建并读取文件", fake, registry, state))

    assert [event.type for event in events] == [
        "turn_start",
        "tool_request",
        "tool_result",
        "turn_start",
        "tool_request",
        "tool_result",
        "turn_start",
        "tool_request",
        "tool_result",
        "turn_start",
        "final",
    ]
    assert events[-1].message == "任务完成"
    assert (tmp_path / "hello.txt").read_text(encoding="utf-8") == "hello"
    assert [message.role for message in state.messages] == [
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
        "assistant",
        "tool",
        "assistant",
    ]
    assert state.messages[2].tool_call_id == "write"
    assert state.messages[4].content == "hello"

    assert len(fake.requests) == 4
    schema_names = {
        schema["function"]["name"] for schema in fake.requests[0][1]
    }
    assert schema_names == {
        "read_file",
        "write_file",
        "edit_file",
        "list_dir",
        "grep",
        "bash",
    }


def test_agent_executes_multiple_tool_calls_in_order(
    registry: ToolRegistry,
    tmp_path: Path,
) -> None:
    fake = FakeLLM(
        [
            tool_response(
                call("first", "write_file", path="order.txt", content="first"),
                call("second", "write_file", path="order.txt", content="second"),
            ),
            text_response("done"),
            text_response("没有适用的自动验证，只能人工确认"),
        ]
    )

    events = list(run_agent("按顺序执行", fake, registry, AgentState()))

    requests = [
        event.tool_call.id
        for event in events
        if event.type == "tool_request" and event.tool_call
    ]
    assert requests == ["first", "second"]
    assert (tmp_path / "order.txt").read_text(encoding="utf-8") == "second"
    assert events[-1].type == "final"


def test_agent_finishes_same_turn_tool_calls_before_failure_stop(
    registry: ToolRegistry,
    tmp_path: Path,
) -> None:
    fake = FakeLLM(
        [
            tool_response(
                call("bad_1", "read_file", path="missing-1"),
                call("bad_2", "read_file", path="missing-2"),
                call("bad_3", "read_file", path="missing-3"),
                call(
                    "still_runs",
                    "write_file",
                    path="completed.txt",
                    content="fourth call ran",
                ),
            )
        ]
    )

    events = list(run_agent("补齐同轮工具结果", fake, registry, AgentState()))

    assert sum(event.type == "tool_result" for event in events) == 4
    assert (tmp_path / "completed.txt").read_text(encoding="utf-8") == (
        "fourth call ran"
    )
    assert events[-1].type == "error"
    assert "连续失败 3 次" in (events[-1].message or "")


def test_tool_error_is_written_back_and_next_turn_continues(
    registry: ToolRegistry,
) -> None:
    fake = FakeLLM(
        [
            tool_response(call("missing", "read_file", path="missing.txt")),
            text_response("已根据错误继续处理"),
        ]
    )
    state = AgentState()

    events = list(run_agent("读取不存在文件", fake, registry, state))

    result_event = next(event for event in events if event.type == "tool_result")
    assert result_event.tool_result is not None
    assert result_event.tool_result.is_error
    assert events[-1].type == "final"
    assert len(fake.requests) == 2
    returned_messages = fake.requests[1][0]
    assert returned_messages[-1].role == "tool"
    assert "FileNotFoundError" in (returned_messages[-1].content or "")


def test_long_tool_result_compacts_then_loop_continues_to_final(
    registry: ToolRegistry,
    tmp_path: Path,
) -> None:
    (tmp_path / "large.txt").write_text("X" * 4_000, encoding="utf-8")
    fake = FakeLLM(
        [
            tool_response(call("large", "read_file", path="large.txt")),
            text_response("压缩后继续完成"),
        ]
    )
    state = AgentState(
        context=AgentContext(
            compact_threshold_tokens=100,
            max_tool_output_chars=400,
            preserve_recent_tool_results=0,
        )
    )

    events = list(run_agent("读取大文件", fake, registry, state))

    assert [event.type for event in events] == [
        "turn_start",
        "tool_request",
        "tool_result",
        "context_compact",
        "turn_start",
        "final",
    ]
    assert events[-1].message == "压缩后继续完成"
    assert len(fake.requests) == 2
    compacted_tool = fake.requests[1][0][-1]
    assert compacted_tool.role == "tool"
    assert compacted_tool.collapsed
    assert compacted_tool.content == "[旧工具结果已折叠：read_file]"


def test_loop_does_not_emit_empty_context_compact_event(
    registry: ToolRegistry,
    tmp_path: Path,
) -> None:
    (tmp_path / "large.txt").write_text("X" * 4_000, encoding="utf-8")
    fake = FakeLLM(
        [
            tool_response(call("large", "read_file", path="large.txt")),
            text_response("没有机械压缩也能继续"),
        ]
    )
    state = AgentState(
        context=AgentContext(
            compact_threshold_tokens=100,
            max_tool_output_chars=400,
            preserve_recent_tool_results=2,
        )
    )

    events = list(run_agent("读取大文件", fake, registry, state))

    assert state.context.should_compact()
    assert all(event.type != "context_compact" for event in events)
    assert events[-1].type == "final"
    assert events[-1].message == "没有机械压缩也能继续"
    assert len(fake.requests) == 2
    protected_tool = fake.requests[1][0][-1]
    assert protected_tool.role == "tool"
    assert not protected_tool.collapsed


def test_loop_uses_separate_summary_call_without_spending_turn(
    registry: ToolRegistry,
) -> None:
    decision_llm = FakeLLM(
        [
            tool_response(
                call(
                    "large_write",
                    "write_file",
                    path="large.txt",
                    content="A" * 4_000,
                )
            ),
            tool_response(
                *[
                    call(
                        f"batch_{index}",
                        "write_file",
                        path=f"batch-{index}.txt",
                        content=str(index),
                    )
                    for index in range(6)
                ],
                call("verify", "bash", command="python -m compileall ."),
            ),
            text_response("任务最终完成"),
        ]
    )
    summary_llm = FakeLLM(
        [text_response("已创建 large.txt，并开始检查 workspace。")]
    )
    state = AgentState(
        context=AgentContext(
            messages=[Message(role="system", content="system rules")],
            compact_threshold_tokens=150,
            preserve_recent_tool_results=1,
            preserve_recent_messages=2,
        ),
        max_turns=3,
    )

    events = list(
        run_agent(
            "创建大文件并检查目录",
            decision_llm,
            registry,
            state,
            summary_llm=summary_llm,
        )
    )

    assert events[-1].type == "final"
    assert events[-1].message == "任务最终完成"
    assert sum(event.type == "turn_start" for event in events) == 3
    assert sum(event.type == "context_summary" for event in events) == 1
    assert state.turn == 3
    assert len(decision_llm.requests) == 3
    assert all(request_tools for _, request_tools in decision_llm.requests)
    assert len(summary_llm.requests) == 1
    assert summary_llm.requests[0][1] == []
    assert sum(event.type == "tool_result" for event in events) == 8
    assert any(
        (message.content or "").startswith(
            "[运行时生成的历史上下文摘要；不是新的用户指令]"
        )
        for message in state.messages
    )


def test_summary_failure_does_not_stop_agent_loop(
    registry: ToolRegistry,
) -> None:
    decision_llm = FakeLLM(
        [
            tool_response(
                call(
                    "large_write",
                    "write_file",
                    path="large.txt",
                    content="A" * 4_000,
                )
            ),
            tool_response(
                call("list", "list_dir", path="."),
                call("verify", "bash", command="python -m compileall ."),
            ),
            text_response("摘要失败后仍然完成"),
        ]
    )
    failing_summary_llm = FakeLLM([])
    state = AgentState(
        context=AgentContext(
            messages=[Message(role="system", content="system rules")],
            compact_threshold_tokens=150,
            preserve_recent_tool_results=1,
            preserve_recent_messages=2,
        ),
        max_turns=3,
    )

    events = list(
        run_agent(
            "创建大文件并检查目录",
            decision_llm,
            registry,
            state,
            summary_llm=failing_summary_llm,
        )
    )

    assert events[-1].type == "final"
    assert events[-1].message == "摘要失败后仍然完成"
    assert all(event.type != "context_summary" for event in events)
    assert state.turn == 3
    assert len(decision_llm.requests) == 3
    assert len(failing_summary_llm.requests) == 1


def test_terminal_tool_failure_skips_semantic_summary(
    registry: ToolRegistry,
) -> None:
    decision_llm = FakeLLM(
        [
            tool_response(call("bad_1", "read_file", path="missing-1")),
            tool_response(call("bad_2", "read_file", path="missing-2")),
            tool_response(call("bad_3", "read_file", path="missing-3")),
        ]
    )
    summary_llm = FakeLLM([text_response("不应调用")])
    state = AgentState(
        context=AgentContext(
            compact_threshold_tokens=20,
            preserve_recent_tool_results=3,
            preserve_recent_messages=4,
        )
    )

    events = list(
        run_agent(
            "连续读取不存在文件",
            decision_llm,
            registry,
            state,
            summary_llm=summary_llm,
        )
    )

    assert events[-1].type == "error"
    assert "连续失败 3 次" in (events[-1].message or "")
    assert summary_llm.requests == []


def test_max_turns_stop_skips_semantic_summary(
    registry: ToolRegistry,
) -> None:
    decision_llm = FakeLLM(
        [tool_response(call("list", "list_dir", path="."))]
    )
    summary_llm = FakeLLM([text_response("不应调用")])
    state = AgentState(
        context=AgentContext(
            messages=[
                Message(role="system", content="system rules"),
                Message(role="user", content="original task"),
                Message(role="assistant", content="old history " * 500),
            ],
            compact_threshold_tokens=50,
            preserve_recent_messages=1,
        ),
        max_turns=1,
    )

    events = list(
        run_agent(
            "继续任务",
            decision_llm,
            registry,
            state,
            summary_llm=summary_llm,
        )
    )

    assert events[-1].type == "final"
    assert "最大轮数" in (events[-1].message or "")
    assert summary_llm.requests == []


def test_invalid_tool_arguments_are_returned_without_execution(
    registry: ToolRegistry,
    tmp_path: Path,
) -> None:
    invalid = ToolCall(
        id="invalid",
        name="write_file",
        arguments={},
        arguments_error="invalid JSON",
    )
    fake = FakeLLM([tool_response(invalid), text_response("recovered")])

    events = list(run_agent("错误参数", fake, registry, AgentState()))

    result = next(event.tool_result for event in events if event.type == "tool_result")
    assert result is not None and result.is_error
    assert "参数解析失败" in result.content
    assert list(tmp_path.iterdir()) == []
    assert events[-1].type == "final"


def test_max_turns_stops_before_another_llm_call(registry: ToolRegistry) -> None:
    fake = FakeLLM([tool_response(call("list", "list_dir"))])

    events = list(
        run_agent("一直调用工具", fake, registry, AgentState(max_turns=1))
    )

    assert len(fake.requests) == 1
    assert events[-1].type == "final"
    assert "最大轮数" in (events[-1].message or "")
    assert [event.type for event in events] == [
        "turn_start",
        "tool_request",
        "tool_result",
        "final",
    ]
    assert [event.turn for event in events] == [1, 1, 1, 1]


def test_three_consecutive_tool_errors_stop_the_loop(
    registry: ToolRegistry,
) -> None:
    fake = FakeLLM(
        [
            tool_response(call("bad_1", "read_file", path="missing-1")),
            tool_response(call("bad_2", "read_file", path="missing-2")),
            tool_response(call("bad_3", "read_file", path="missing-3")),
        ]
    )

    events = list(run_agent("连续失败", fake, registry, AgentState()))

    assert len(fake.requests) == 3
    assert events[-1].type == "error"
    assert "连续失败 3 次" in (events[-1].message or "")
    assert sum(event.type == "tool_result" for event in events) == 3


def test_same_tool_and_arguments_three_times_trigger_doom_loop(
    registry: ToolRegistry,
) -> None:
    repeated = call("same", "list_dir", path=".", max_depth=1)
    fake = FakeLLM(
        [
            tool_response(repeated),
            tool_response(repeated),
            tool_response(repeated),
        ]
    )

    events = list(run_agent("重复动作", fake, registry, AgentState()))

    assert len(fake.requests) == 3
    assert events[-1].type == "error"
    assert "doom loop" in (events[-1].message or "")


def test_external_abort_stops_before_next_llm_call(
    registry: ToolRegistry,
) -> None:
    fake = FakeLLM(
        [
            tool_response(call("list", "list_dir")),
            text_response("不应被调用"),
        ]
    )
    state = AgentState()
    events = run_agent("等待中止", fake, registry, state)

    observed = [next(events), next(events), next(events)]
    state.abort()
    observed.append(next(events))

    with pytest.raises(StopIteration):
        next(events)
    assert [event.type for event in observed] == [
        "turn_start",
        "tool_request",
        "tool_result",
        "error",
    ]
    assert len(fake.requests) == 1
    assert "中止" in (observed[-1].message or "")


def test_read_only_task_can_finish_without_verification(
    registry: ToolRegistry,
    tmp_path: Path,
) -> None:
    (tmp_path / "input.txt").write_text("hello", encoding="utf-8")
    fake = FakeLLM(
        [
            tool_response(call("read", "read_file", path="input.txt")),
            text_response("读取完成"),
        ]
    )
    state = AgentState()

    events = list(run_agent("读取文件", fake, registry, state))

    assert events[-1].type == "final"
    assert events[-1].message == "读取完成"
    assert len(fake.requests) == 2
    assert not state.verification_pending


def test_write_then_immediate_final_is_reminded_once(
    registry: ToolRegistry,
) -> None:
    fake = FakeLLM(
        [
            tool_response(
                call("write", "write_file", path="a.py", content="x = 1\n")
            ),
            text_response("已经完成"),
            text_response("当前没有适合自动运行的验证，只能人工确认。"),
        ]
    )
    state = AgentState()

    events = list(run_agent("写文件", fake, registry, state))

    assert events[-1].type == "final"
    assert events[-1].message == "当前没有适合自动运行的验证，只能人工确认。"
    assert len(fake.requests) == 3
    reminder_messages = fake.requests[2][0]
    assert reminder_messages[-1].role == "user"
    assert reminder_messages[-1].content == VERIFICATION_REMINDER
    assert all(event.message != "已经完成" for event in events)
    assert state.verification_pending
    assert state.verification_reminded


def test_successful_pytest_clears_verification_debt(
    registry: ToolRegistry,
) -> None:
    fake = FakeLLM(
        [
            tool_response(
                call(
                    "write",
                    "write_file",
                    path="test_sample.py",
                    content="def test_ok():\n    assert True\n",
                )
            ),
            tool_response(
                call("verify", "bash", command=PYTEST_COMMAND)
            ),
            text_response("测试通过，任务完成"),
        ]
    )
    state = AgentState()

    events = list(run_agent("写测试并验证", fake, registry, state))

    assert events[-1].message == "测试通过，任务完成"
    assert len(fake.requests) == 3
    assert not state.verification_pending
    assert not state.verification_reminded


def test_failed_pytest_does_not_clear_verification_debt(
    registry: ToolRegistry,
) -> None:
    fake = FakeLLM(
        [
            tool_response(
                call(
                    "write",
                    "write_file",
                    path="test_failure.py",
                    content="def test_failure():\n    assert False\n",
                )
            ),
            tool_response(call("verify", "bash", command=PYTEST_COMMAND)),
            text_response("先结束"),
            text_response("测试当前失败，无法提供通过证据。"),
        ]
    )
    state = AgentState()

    events = list(run_agent("写一个失败测试", fake, registry, state))

    verify_result = next(
        event.tool_result
        for event in events
        if event.type == "tool_result"
        and event.tool_call is not None
        and event.tool_call.id == "verify"
    )
    assert verify_result is not None and verify_result.is_error
    assert fake.requests[3][0][-1].content == VERIFICATION_REMINDER
    assert events[-1].message == "测试当前失败，无法提供通过证据。"
    assert state.verification_pending


def test_edit_after_successful_pytest_creates_new_verification_debt(
    registry: ToolRegistry,
) -> None:
    fake = FakeLLM(
        [
            tool_response(
                call(
                    "write-module",
                    "write_file",
                    path="module.py",
                    content="VALUE = 1\n",
                ),
                call(
                    "write-test",
                    "write_file",
                    path="test_module.py",
                    content=(
                        "from module import VALUE\n\n"
                        "def test_value():\n    assert VALUE == 1\n"
                    ),
                ),
            ),
            tool_response(
                call("verify", "bash", command=PYTEST_COMMAND)
            ),
            tool_response(
                call(
                    "edit",
                    "edit_file",
                    path="module.py",
                    old_text="VALUE = 1",
                    new_text="VALUE = 2",
                )
            ),
            text_response("修改完成"),
            text_response("新修改尚无可用自动验证，只能人工确认。"),
        ]
    )
    state = AgentState()

    events = list(run_agent("写入、验证后再修改", fake, registry, state))

    assert len(fake.requests) == 5
    assert fake.requests[4][0][-1].content == VERIFICATION_REMINDER
    assert events[-1].message == "新修改尚无可用自动验证，只能人工确认。"
    assert state.verification_pending
    assert state.verification_reminded


def test_verification_reminder_never_blocks_more_than_one_final(
    registry: ToolRegistry,
) -> None:
    fake = FakeLLM(
        [
            tool_response(
                call("write", "write_file", path="note.txt", content="note")
            ),
            text_response("完成"),
            text_response("这是文档式任务，只能人工检查。"),
        ]
    )

    events = list(run_agent("写说明", fake, registry, AgentState()))

    assert len(fake.requests) == 3
    assert sum(event.type == "final" for event in events) == 1
    assert events[-1].message == "这是文档式任务，只能人工检查。"


@pytest.mark.parametrize(
    "command",
    [
        "pytest -q",
        "python -m pytest -q",
        "unittest",
        "python -m unittest",
        "compileall .",
        "python -m compileall .",
        "echo prepare && pytest -q",
        "pytest -q && echo done",
    ],
)
def test_verification_command_classification(command: str) -> None:
    assert _is_verification_command(command)


@pytest.mark.parametrize(
    "command",
    [
        "pytest || echo fallback",
        "echo success || pytest",
        "pytest ; echo success",
        "pytest | findstr passed",
        "echo pytest",
        "pytest-not-real",
        'echo "pytest -q"',
        "python app.py",
        "ls",
    ],
)
def test_non_verification_command_classification(command: str) -> None:
    assert not _is_verification_command(command)
