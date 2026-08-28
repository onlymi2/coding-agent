from pathlib import Path

import pytest

from ke.agent.loop import AgentState, run_agent
from ke.llm.fake_llm import FakeLLM
from ke.llm.types import LLMResponse, Message, ToolCall
from ke.safety.sandbox import WorkspaceSandbox
from ke.tools.registry import ToolRegistry


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
    ]
    assert state.messages[2].tool_call_id == "write"
    assert state.messages[4].content == "hello"

    assert len(fake.requests) == 3
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
