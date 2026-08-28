import threading
from pathlib import Path

import pytest

from ke.agent.loop import AgentState, run_agent
from ke.llm.fake_llm import FakeLLM
from ke.llm.types import LLMResponse, Message, ToolCall
from ke.safety.confirm import (
    PermissionAlreadyResolvedError,
    PermissionGate,
    UnknownPermissionError,
)
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


def test_permission_rules_and_bounded_previews() -> None:
    gate = PermissionGate()

    for name in ("read_file", "list_dir", "grep"):
        assert not gate.requires_confirmation(ToolCall("id", name, {}))
    for name in ("write_file", "edit_file", "bash"):
        assert gate.requires_confirmation(ToolCall("id", name, {}))

    write_request = gate.create(
        ToolCall(
            "write",
            "write_file",
            {"path": "safe.txt", "content": "SECRET" * 100},
        )
    )
    bash_request = gate.create(
        ToolCall("bash", "bash", {"command": "x" * 500})
    )

    assert write_request.preview == "safe.txt"
    assert "SECRET" not in write_request.preview
    assert len(bash_request.preview) == 200


def test_permission_wait_resolve_and_errors() -> None:
    gate = PermissionGate()
    request = gate.create(ToolCall("write", "write_file", {"path": "x"}))
    result: list[bool] = []
    waiter = threading.Thread(
        target=lambda: result.append(gate.wait(request.permission_id))
    )
    waiter.start()

    gate.resolve(request.permission_id, True)
    waiter.join(timeout=2)

    assert not waiter.is_alive()
    assert result == [True]
    with pytest.raises(PermissionAlreadyResolvedError):
        gate.resolve(request.permission_id, False)
    with pytest.raises(UnknownPermissionError):
        gate.resolve("missing", True)


def test_cancel_latch_rejects_permissions_created_after_cancel() -> None:
    gate = PermissionGate()
    gate.cancel_all()

    late = gate.create(
        ToolCall("late", "write_file", {"path": "late.txt"})
    )

    assert gate.wait(late.permission_id) is False
    with pytest.raises(PermissionAlreadyResolvedError):
        gate.resolve(late.permission_id, True)

    gate.reset()
    next_run = gate.create(
        ToolCall("next", "write_file", {"path": "next.txt"})
    )
    gate.resolve(next_run.permission_id, True)
    assert gate.wait(next_run.permission_id) is True


def test_confirm_event_is_emitted_before_wait_and_allow_executes(
    tmp_path: Path,
) -> None:
    gate = PermissionGate()
    fake = FakeLLM(
        [
            tool_response(
                ToolCall(
                    "write",
                    "write_file",
                    {"path": "allowed.txt", "content": "allowed"},
                )
            ),
            text_response("done"),
            text_response("没有适用的自动验证，只能人工确认"),
        ]
    )
    events = run_agent(
        "write",
        fake,
        ToolRegistry(WorkspaceSandbox(tmp_path)),
        AgentState(),
        confirmation=gate,
    )

    observed = [next(events), next(events), next(events)]
    confirm = observed[-1]
    assert [event.type for event in observed] == [
        "turn_start",
        "tool_request",
        "tool_confirm",
    ]
    assert not (tmp_path / "allowed.txt").exists()

    gate.resolve(confirm.permission_id or "", True)
    observed.extend(list(events))

    assert (tmp_path / "allowed.txt").read_text(encoding="utf-8") == "allowed"
    assert observed[-1].type == "final"


def test_abort_between_tool_request_and_confirmation_never_waits_or_executes(
    tmp_path: Path,
) -> None:
    gate = PermissionGate()
    state = AgentState()
    events = run_agent(
        "write",
        FakeLLM(
            [
                tool_response(
                    ToolCall(
                        "write",
                        "write_file",
                        {"path": "race.txt", "content": "must not exist"},
                    )
                )
            ]
        ),
        ToolRegistry(WorkspaceSandbox(tmp_path)),
        state,
        confirmation=gate,
    )

    assert next(events).type == "turn_start"
    assert next(events).type == "tool_request"
    state.abort()
    gate.cancel_all()
    remaining = list(events)

    assert [event.type for event in remaining] == ["error"]
    assert not (tmp_path / "race.txt").exists()


def test_denied_tool_is_not_executed_and_error_returns_to_model(
    tmp_path: Path,
) -> None:
    gate = PermissionGate()
    fake = FakeLLM(
        [
            tool_response(
                ToolCall(
                    "write",
                    "write_file",
                    {"path": "denied.txt", "content": "denied"},
                )
            ),
            text_response("adjusted after refusal"),
        ]
    )
    state = AgentState()
    events = run_agent(
        "write",
        fake,
        ToolRegistry(WorkspaceSandbox(tmp_path)),
        state,
        confirmation=gate,
    )

    observed = [next(events), next(events), next(events)]
    gate.resolve(observed[-1].permission_id or "", False)
    observed.extend(list(events))

    assert not (tmp_path / "denied.txt").exists()
    result_event = next(event for event in observed if event.type == "tool_result")
    assert result_event.tool_result is not None
    assert result_event.tool_result.is_error
    assert "用户拒绝执行工具：write_file" in result_event.tool_result.content
    assert fake.requests[1][0][-1].role == "tool"
    assert "用户拒绝" in (fake.requests[1][0][-1].content or "")
    assert observed[-1].type == "final"


@pytest.mark.parametrize(
    "call",
    [
        ToolCall("write", "write_file", {"path": "x", "content": "x"}),
        ToolCall("edit", "edit_file", {"path": "x", "old_text": "a", "new_text": "b"}),
        ToolCall("bash", "bash", {"command": "echo x"}),
    ],
)
def test_each_dangerous_tool_emits_confirm(
    tmp_path: Path,
    call: ToolCall,
) -> None:
    gate = PermissionGate()
    fake = FakeLLM([tool_response(call), text_response("done")])
    events = run_agent(
        "dangerous",
        fake,
        ToolRegistry(WorkspaceSandbox(tmp_path)),
        AgentState(),
        confirmation=gate,
    )

    observed = [next(events), next(events), next(events)]
    assert observed[-1].type == "tool_confirm"
    assert observed[-1].tool_call == call
    gate.resolve(observed[-1].permission_id or "", False)
    list(events)


def test_safe_tools_do_not_emit_confirm(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("needle", encoding="utf-8")
    fake = FakeLLM(
        [
            tool_response(
                ToolCall("read", "read_file", {"path": "a.txt"}),
                ToolCall("list", "list_dir", {"path": "."}),
                ToolCall("grep", "grep", {"pattern": "needle"}),
            ),
            text_response("done"),
        ]
    )

    events = list(
        run_agent(
            "safe tools",
            fake,
            ToolRegistry(WorkspaceSandbox(tmp_path)),
            AgentState(),
            confirmation=PermissionGate(),
        )
    )

    assert all(event.type != "tool_confirm" for event in events)
    assert sum(event.type == "tool_result" for event in events) == 3


def test_auto_approve_executes_without_confirm(tmp_path: Path) -> None:
    fake = FakeLLM(
        [
            tool_response(
                ToolCall(
                    "write",
                    "write_file",
                    {"path": "auto.txt", "content": "auto"},
                )
            ),
            text_response("done"),
            text_response("没有适用的自动验证，只能人工确认"),
        ]
    )

    events = list(
        run_agent(
            "auto",
            fake,
            ToolRegistry(WorkspaceSandbox(tmp_path)),
            AgentState(),
            confirmation=PermissionGate(auto_approve=True),
        )
    )

    assert all(event.type != "tool_confirm" for event in events)
    assert (tmp_path / "auto.txt").read_text(encoding="utf-8") == "auto"


def test_mixed_tool_batch_keeps_order_across_multiple_confirmations(
    tmp_path: Path,
) -> None:
    (tmp_path / "source.txt").write_text("needle", encoding="utf-8")
    calls = [
        ToolCall("read", "read_file", {"path": "source.txt"}),
        ToolCall(
            "write",
            "write_file",
            {"path": "created.txt", "content": "created"},
        ),
        ToolCall("grep", "grep", {"pattern": "needle"}),
        ToolCall("bash", "bash", {"command": "echo should-not-run"}),
    ]
    fake = FakeLLM(
        [
            tool_response(*calls),
            text_response("done"),
            text_response("没有适用的自动验证，只能人工确认"),
        ]
    )
    gate = PermissionGate()
    event_iterator = run_agent(
        "mixed",
        fake,
        ToolRegistry(WorkspaceSandbox(tmp_path)),
        AgentState(),
        confirmation=gate,
    )
    observed = []

    while True:
        event = next(event_iterator)
        observed.append(event)
        if event.type == "tool_confirm":
            assert event.permission_id is not None
            gate.resolve(
                event.permission_id,
                event.tool_call is not None
                and event.tool_call.name == "write_file",
            )
        if event.type == "final":
            break

    assert [
        event.tool_call.name
        for event in observed
        if event.type == "tool_request" and event.tool_call is not None
    ] == [call.name for call in calls]
    assert [
        event.tool_call.name
        for event in observed
        if event.type == "tool_result" and event.tool_call is not None
    ] == [call.name for call in calls]
    assert [
        event.tool_call.name
        for event in observed
        if event.type == "tool_confirm" and event.tool_call is not None
    ] == ["write_file", "bash"]
    assert (tmp_path / "created.txt").read_text(encoding="utf-8") == "created"
    assert len([message for message in fake.requests[1][0] if message.role == "tool"]) == 4
