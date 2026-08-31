import json
import threading
import time
from collections.abc import Iterator, Sequence
from pathlib import Path

from starlette.testclient import TestClient

from ke.config import KeConfig
from ke.llm.fake_llm import FakeLLM
from ke.llm.protocol import ToolSchema
from ke.llm.types import LLMResponse, Message, ToolCall
from ke.server.app import AgentSession, StoredEvent, _event_stream, create_app


def text_response(content: str) -> LLMResponse:
    return LLMResponse(
        message=Message(role="assistant", content=content),
        finish_reason="stop",
    )


def tool_response(*calls: ToolCall) -> LLMResponse:
    return LLMResponse(
        message=Message(role="assistant", tool_calls=list(calls)),
        finish_reason="tool_calls",
    )


def make_config(tmp_path: Path, max_turns: int = 10) -> KeConfig:
    return KeConfig(
        channel="test",
        base_url="https://unused.example/v1",
        model="fake-model",
        api_key="test-key",
        workspace=tmp_path,
        max_turns=max_turns,
        max_tool_output_chars=8_000,
        compact_threshold_tokens=24_000,
    )


def create_session(client: TestClient) -> str:
    response = client.post("/session")
    assert response.status_code == 201
    return response.json()["id"]


def session_for(app: object, session_id: str) -> AgentSession:
    session = app.state.runtime.get(session_id)
    assert session is not None
    return session


def wait_for_event(
    session: AgentSession,
    event_type: str,
    timeout: float = 3.0,
) -> StoredEvent:
    deadline = time.monotonic() + timeout
    cursor = 0
    while time.monotonic() < deadline:
        events = session.events.snapshot()
        for event in events:
            if event.event_type == event_type:
                return event
        cursor = len(events)
        session.events.wait_after(cursor, timeout=0.05)
    raise AssertionError(f"timed out waiting for event {event_type}")


def parse_sse(text: str) -> list[dict[str, object]]:
    parsed: list[dict[str, object]] = []
    for block in text.replace("\r\n", "\n").split("\n\n"):
        if not block or block.startswith(":"):
            continue
        fields: dict[str, str] = {}
        for line in block.splitlines():
            name, value = line.split(":", 1)
            fields[name] = value.strip()
        parsed.append(
            {
                "id": int(fields["id"]),
                "event": fields["event"],
                "data": json.loads(fields["data"]),
            }
        )
    return parsed


def consume_sse(
    stream: Iterator[str],
    collected: list[dict[str, object]],
    *,
    terminal_count: int,
    done: threading.Event,
    first_terminal: threading.Event | None = None,
) -> None:
    terminals = 0
    for frame in stream:
        for event in parse_sse(frame):
            collected.append(event)
            if event["event"] in {"final", "error"}:
                terminals += 1
                if terminals == 1 and first_terminal is not None:
                    first_terminal.set()
                if terminals >= terminal_count:
                    done.set()
                    return


class BlockingLLM:
    def __init__(self, response: LLMResponse) -> None:
        self.response = response
        self.started = threading.Event()
        self.release = threading.Event()
        self.requests: list[tuple[list[Message], list[ToolSchema]]] = []

    def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSchema],
    ) -> LLMResponse:
        self.requests.append((list(messages), list(tools)))
        self.started.set()
        if not self.release.wait(timeout=5):
            raise RuntimeError("test did not release blocking LLM")
        return self.response


class FailingLLM:
    def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSchema],
    ) -> LLMResponse:
        raise RuntimeError("sdk-secret-detail")


def test_health_and_sessions_are_isolated(tmp_path: Path) -> None:
    factory_calls: list[int] = []

    def factory() -> FakeLLM:
        factory_calls.append(1)
        return FakeLLM([text_response("done")])

    app = create_app(make_config(tmp_path), llm_factory=factory)
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json() == {"status": "ok"}
        assert factory_calls == []

        first_id = create_session(client)
        second_id = create_session(client)
        first = session_for(app, first_id)
        second = session_for(app, second_id)

        assert first.id != second.id
        assert first.state is not second.state
        assert first.state.context is not second.state.context
        assert first.permissions is not second.permissions
        assert first.events is not second.events
        assert first.state.messages[0].role == "system"
        assert second.state.messages[0].role == "system"

        assert client.post(f"/session/{first_id}/abort").status_code == 202
        assert first.state.cancelled
        assert not second.state.cancelled
        assert client.post("/session/missing/message", json={"content": "x"}).status_code == 404
        assert len(app.routes) == 7
        assert {route.path for route in app.routes} == {
            "/",
            "/health",
            "/session",
            "/session/{id}/message",
            "/session/{id}/events",
            "/session/{id}/permissions/{pid}",
            "/session/{id}/abort",
        }


def test_message_returns_202_while_worker_is_blocked(tmp_path: Path) -> None:
    blocking = BlockingLLM(text_response("done"))
    app = create_app(make_config(tmp_path), llm_factory=lambda: blocking)
    with TestClient(app) as client:
        session_id = create_session(client)
        session = session_for(app, session_id)

        response = client.post(
            f"/session/{session_id}/message",
            json={"content": "wait"},
        )

        assert response.status_code == 202
        assert blocking.started.wait(timeout=1)
        assert session.running
        duplicate = client.post(
            f"/session/{session_id}/message",
            json={"content": "duplicate"},
        )
        assert duplicate.status_code == 409

        blocking.release.set()
        assert session.wait_until_idle()
        assert not session.running


def test_message_validation_and_running_resets_after_final(tmp_path: Path) -> None:
    app = create_app(
        make_config(tmp_path),
        llm_factory=lambda: FakeLLM([text_response("done")]),
    )
    with TestClient(app) as client:
        session_id = create_session(client)
        assert client.post(
            f"/session/{session_id}/message", json={"content": ""}
        ).status_code == 400
        assert client.post(
            f"/session/{session_id}/message", json={"content": 123}
        ).status_code == 400

        assert client.post(
            f"/session/{session_id}/message", json={"content": "finish"}
        ).status_code == 202
        session = session_for(app, session_id)
        assert session.wait_until_idle()
        assert not session.running


def test_idle_sse_subscriber_waits_for_later_task(tmp_path: Path) -> None:
    app = create_app(
        make_config(tmp_path),
        llm_factory=lambda: FakeLLM([text_response("done")]),
    )
    with TestClient(app) as client:
        session_id = create_session(client)
        session = session_for(app, session_id)
        entered_wait = threading.Event()
        original_wait = session.events.wait_after

        def observed_wait(after_id: int, timeout: float = 15.0) -> list[StoredEvent]:
            entered_wait.set()
            return original_wait(after_id, timeout=timeout)

        session.events.wait_after = observed_wait  # type: ignore[method-assign]
        stream = _event_stream(session, 0)
        collected: list[dict[str, object]] = []
        done = threading.Event()
        subscriber = threading.Thread(
            target=consume_sse,
            args=(stream, collected),
            kwargs={"terminal_count": 1, "done": done},
            daemon=True,
        )
        subscriber.start()

        assert entered_wait.wait(timeout=1)
        assert not done.wait(timeout=0.05)
        assert subscriber.is_alive()
        assert client.post(
            f"/session/{session_id}/message",
            json={"content": "start after attach"},
        ).status_code == 202
        assert done.wait(timeout=3)
        assert session.wait_until_idle()

        stream.close()
        subscriber.join(timeout=1)
        assert not subscriber.is_alive()
        assert [event["event"] for event in collected] == [
            "turn_start",
            "final",
        ]


def test_sse_broadcasts_same_task_to_two_idle_subscribers(tmp_path: Path) -> None:
    app = create_app(
        make_config(tmp_path),
        llm_factory=lambda: FakeLLM(
            [
                tool_response(ToolCall("list", "list_dir", {"path": "."})),
                text_response("done"),
            ]
        ),
    )
    with TestClient(app) as client:
        session_id = create_session(client)
        session = session_for(app, session_id)
        streams = [_event_stream(session, 0), _event_stream(session, 0)]
        collected: list[list[dict[str, object]]] = [[], []]
        done = [threading.Event(), threading.Event()]
        subscribers = [
            threading.Thread(
                target=consume_sse,
                args=(streams[index], collected[index]),
                kwargs={"terminal_count": 1, "done": done[index]},
                daemon=True,
            )
            for index in range(2)
        ]
        for subscriber in subscribers:
            subscriber.start()

        assert client.post(
            f"/session/{session_id}/message",
            json={"content": "list files"},
        ).status_code == 202
        assert all(signal.wait(timeout=3) for signal in done)
        assert session.wait_until_idle()
        for stream in streams:
            stream.close()
        for subscriber in subscribers:
            subscriber.join(timeout=1)
            assert not subscriber.is_alive()

        first, second = collected
        assert [event["event"] for event in first] == [
            "turn_start",
            "tool_request",
            "tool_result",
            "turn_start",
            "final",
        ]
        assert [event["id"] for event in first] == list(
            range(1, len(first) + 1)
        )
        assert first == second
        tool_request = next(
            event for event in first if event["event"] == "tool_request"
        )
        assert tool_request["data"]["tool_call"] == {
            "id": "list",
            "name": "list_dir",
            "arguments": {"path": "."},
            "arguments_error": None,
        }
        tool_result = next(
            event for event in first if event["event"] == "tool_result"
        )
        assert set(tool_result["data"]["tool_result"]) == {
            "content",
            "is_error",
        }
        assert tool_result["data"]["tool_result"]["is_error"] is False
        assert isinstance(first[-1]["data"], dict)


def test_sse_observer_continues_across_two_tasks_in_one_session(
    tmp_path: Path,
) -> None:
    app = create_app(
        make_config(tmp_path),
        llm_factory=lambda: FakeLLM(
            [text_response("task A done"), text_response("task B done")]
        ),
    )
    with TestClient(app) as client:
        session_id = create_session(client)
        session = session_for(app, session_id)
        stream = _event_stream(session, 0)
        collected: list[dict[str, object]] = []
        first_terminal = threading.Event()
        done = threading.Event()
        subscriber = threading.Thread(
            target=consume_sse,
            args=(stream, collected),
            kwargs={
                "terminal_count": 2,
                "done": done,
                "first_terminal": first_terminal,
            },
            daemon=True,
        )
        subscriber.start()

        assert client.post(
            f"/session/{session_id}/message",
            json={"content": "task A"},
        ).status_code == 202
        assert first_terminal.wait(timeout=3)
        assert session.wait_until_idle()
        assert subscriber.is_alive()

        assert client.post(
            f"/session/{session_id}/message",
            json={"content": "task B"},
        ).status_code == 202
        assert done.wait(timeout=3)
        assert session.wait_until_idle()

        stream.close()
        subscriber.join(timeout=1)
        assert not subscriber.is_alive()
        assert [event["event"] for event in collected] == [
            "turn_start",
            "final",
            "turn_start",
            "final",
        ]
        assert [event["id"] for event in collected] == [1, 2, 3, 4]
        assert collected[1]["data"]["message"] == "task A done"
        assert collected[3]["data"]["message"] == "task B done"


def test_sse_replays_events_after_cursor(tmp_path: Path) -> None:
    app = create_app(
        make_config(tmp_path),
        llm_factory=lambda: FakeLLM(
            [
                tool_response(ToolCall("list", "list_dir", {"path": "."})),
                text_response("done"),
            ]
        ),
    )
    with TestClient(app) as client:
        session_id = create_session(client)
        session = session_for(app, session_id)
        assert client.post(
            f"/session/{session_id}/message",
            json={"content": "list files"},
        ).status_code == 202
        assert session.wait_until_idle()

        stream = _event_stream(session, 3)
        resumed: list[dict[str, object]] = []
        done = threading.Event()
        consume_sse(stream, resumed, terminal_count=1, done=done)
        stream.close()

        assert done.is_set()
        assert [event["id"] for event in resumed] == [4, 5]


def test_server_does_not_expose_llm_exception_details(tmp_path: Path) -> None:
    app = create_app(make_config(tmp_path), llm_factory=FailingLLM)
    with TestClient(app) as client:
        session_id = create_session(client)
        session = session_for(app, session_id)
        assert client.post(
            f"/session/{session_id}/message", json={"content": "fail"}
        ).status_code == 202
        assert session.wait_until_idle()

        stream = _event_stream(session, 0)
        events: list[dict[str, object]] = []
        done = threading.Event()
        consume_sse(stream, events, terminal_count=1, done=done)
        stream.close()

        serialized = json.dumps(events, ensure_ascii=False)
        assert "sdk-secret-detail" not in serialized
        assert events[-1]["event"] == "error"
        assert events[-1]["data"]["message"] == "LLM 调用失败"


def test_permission_allow_writes_only_after_http_answer(tmp_path: Path) -> None:
    app = create_app(
        make_config(tmp_path),
        llm_factory=lambda: FakeLLM(
            [
                tool_response(
                    ToolCall(
                        "write",
                        "write_file",
                        {"path": "allowed.txt", "content": "allowed"},
                    )
                ),
                text_response("done"),
            ]
        ),
    )
    with TestClient(app) as client:
        session_id = create_session(client)
        session = session_for(app, session_id)
        assert client.post(
            f"/session/{session_id}/message", json={"content": "write"}
        ).status_code == 202

        confirm = wait_for_event(session, "tool_confirm")
        permission_id = str(confirm.data["permission_id"])
        assert not (tmp_path / "allowed.txt").exists()

        allowed = client.post(
            f"/session/{session_id}/permissions/{permission_id}",
            json={"allow": True},
        )
        assert allowed.status_code == 200
        assert allowed.json() == {
            "permission_id": permission_id,
            "allowed": True,
        }
        assert client.post(
            f"/session/{session_id}/permissions/{permission_id}",
            json={"allow": False},
        ).status_code == 409

        assert session.wait_until_idle()
        assert (tmp_path / "allowed.txt").read_text(encoding="utf-8") == "allowed"


def test_permission_deny_returns_tool_error_and_continues(tmp_path: Path) -> None:
    fake = FakeLLM(
        [
            tool_response(
                ToolCall(
                    "write",
                    "write_file",
                    {"path": "denied.txt", "content": "denied"},
                )
            ),
            text_response("continued after denial"),
        ]
    )
    app = create_app(make_config(tmp_path), llm_factory=lambda: fake)
    with TestClient(app) as client:
        session_id = create_session(client)
        session = session_for(app, session_id)
        assert client.post(
            f"/session/{session_id}/message", json={"content": "write"}
        ).status_code == 202
        confirm = wait_for_event(session, "tool_confirm")
        permission_id = str(confirm.data["permission_id"])

        assert client.post(
            f"/session/{session_id}/permissions/{permission_id}",
            json={"allow": False},
        ).status_code == 200
        assert session.wait_until_idle()

        assert not (tmp_path / "denied.txt").exists()
        assert len(fake.requests) == 2
        assert "用户拒绝执行工具：write_file" in (
            fake.requests[1][0][-1].content or ""
        )
        assert session.events.snapshot()[-1].event_type == "final"
        assert client.post(
            f"/session/{session_id}/permissions/missing",
            json={"allow": True},
        ).status_code == 404
        assert client.post(
            f"/session/{session_id}/permissions/missing",
            json={"allow": "yes"},
        ).status_code == 400


def test_permission_ids_are_isolated_between_sessions(tmp_path: Path) -> None:
    def factory() -> FakeLLM:
        return FakeLLM(
            [
                tool_response(
                    ToolCall(
                        "write",
                        "write_file",
                        {"path": "isolated.txt", "content": "isolated"},
                    )
                ),
                text_response("done"),
            ]
        )

    app = create_app(make_config(tmp_path), llm_factory=factory)
    with TestClient(app) as client:
        first_id = create_session(client)
        second_id = create_session(client)
        first = session_for(app, first_id)
        assert client.post(
            f"/session/{first_id}/message", json={"content": "write"}
        ).status_code == 202
        confirm = wait_for_event(first, "tool_confirm")
        permission_id = str(confirm.data["permission_id"])

        assert client.post(
            f"/session/{second_id}/permissions/{permission_id}",
            json={"allow": True},
        ).status_code == 404
        assert client.post(
            f"/session/{first_id}/permissions/{permission_id}",
            json={"allow": False},
        ).status_code == 200
        assert first.wait_until_idle()


def test_auto_approve_dangerous_tool_does_not_block(tmp_path: Path) -> None:
    app = create_app(
        make_config(tmp_path),
        llm_factory=lambda: FakeLLM(
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
        ),
        auto_approve=True,
    )
    with TestClient(app) as client:
        session_id = create_session(client)
        session = session_for(app, session_id)
        assert client.post(
            f"/session/{session_id}/message", json={"content": "write"}
        ).status_code == 202
        assert session.wait_until_idle()

        assert (tmp_path / "auto.txt").read_text(encoding="utf-8") == "auto"
        assert all(
            event.event_type != "tool_confirm"
            for event in session.events.snapshot()
        )


def test_abort_releases_pending_permission_and_skips_tool(tmp_path: Path) -> None:
    app = create_app(
        make_config(tmp_path),
        llm_factory=lambda: FakeLLM(
            [
                tool_response(
                    ToolCall(
                        "write",
                        "write_file",
                        {"path": "aborted.txt", "content": "no"},
                    )
                )
            ]
        ),
    )
    with TestClient(app) as client:
        session_id = create_session(client)
        session = session_for(app, session_id)
        assert client.post(
            f"/session/{session_id}/message", json={"content": "write"}
        ).status_code == 202
        confirm = wait_for_event(session, "tool_confirm")
        permission_id = str(confirm.data["permission_id"])

        assert client.post(f"/session/{session_id}/abort").status_code == 202
        assert session.wait_until_idle()
        worker = session.worker
        assert worker is not None
        worker.join(timeout=1)

        assert not worker.is_alive()
        assert not (tmp_path / "aborted.txt").exists()
        assert session.state.cancelled
        assert session.events.snapshot()[-1].event_type == "error"
        assert client.post(
            f"/session/{session_id}/permissions/{permission_id}",
            json={"allow": True},
        ).status_code == 409


def test_abort_during_llm_call_prevents_tools_and_next_call(tmp_path: Path) -> None:
    blocking = BlockingLLM(
        tool_response(ToolCall("list", "list_dir", {"path": "."}))
    )
    app = create_app(make_config(tmp_path), llm_factory=lambda: blocking)
    with TestClient(app) as client:
        session_id = create_session(client)
        session = session_for(app, session_id)
        assert client.post(
            f"/session/{session_id}/message", json={"content": "wait"}
        ).status_code == 202
        assert blocking.started.wait(timeout=1)

        assert client.post(f"/session/{session_id}/abort").status_code == 202
        blocking.release.set()
        assert session.wait_until_idle()

        assert len(blocking.requests) == 1
        event_types = [event.event_type for event in session.events.snapshot()]
        assert event_types == ["turn_start", "error"]
