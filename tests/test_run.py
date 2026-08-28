from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from ke.client.http import SseEvent
from ke.client.run import run_task
from ke.config import KeConfig


def make_config(tmp_path: Path) -> KeConfig:
    return KeConfig(
        channel="test",
        base_url="https://unused.example/v1",
        model="fake-model",
        api_key="test-key",
        workspace=tmp_path,
    )


class FakeServer:
    def __init__(self, app: object) -> None:
        self.app = app
        self.base_url = "http://127.0.0.1:43210"
        self.started = False
        self.stopped = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True


class FakeHttpClient:
    def __init__(self, events: list[SseEvent]) -> None:
        self.event_values = events
        self.created = False
        self.sent: list[tuple[str, str]] = []
        self.permissions: list[tuple[str, str, bool]] = []
        self.aborted: list[str] = []
        self.closed = False

    def create_session(self) -> str:
        self.created = True
        return "session-1"

    def send_message(self, session_id: str, content: str) -> None:
        self.sent.append((session_id, content))

    def listen_events(self, session_id: str) -> Iterator[SseEvent]:
        yield from self.event_values

    def resolve_permission(
        self,
        session_id: str,
        permission_id: str,
        allow: bool,
    ) -> None:
        self.permissions.append((session_id, permission_id, allow))

    def abort(self, session_id: str) -> None:
        self.aborted.append(session_id)

    def close(self) -> None:
        self.closed = True


def event(event_id: int, name: str, **data: Any) -> SseEvent:
    return SseEvent(event_id, name, {"type": name, **data})


def run_with_fakes(
    config: KeConfig,
    events: list[SseEvent],
    *,
    auto_approve: bool = False,
    input_func: Any = input,
) -> tuple[int, list[str], FakeServer, FakeHttpClient, dict[str, object]]:
    outputs: list[str] = []
    captured: dict[str, object] = {}
    client = FakeHttpClient(events)
    server_holder: list[FakeServer] = []

    def app_factory(value: KeConfig, *, auto_approve: bool) -> Any:
        captured.update(config=value, auto_approve=auto_approve)
        return object()

    def server_factory(app: Any) -> FakeServer:
        server = FakeServer(app)
        server_holder.append(server)
        return server

    code = run_task(
        config,
        "task",
        auto_approve=auto_approve,
        output=outputs.append,
        input_func=input_func,
        app_factory=app_factory,
        server_factory=server_factory,
        http_client_factory=lambda url: client,
    )
    return code, outputs, server_holder[0], client, captured


def test_normal_run_prints_plain_events_and_returns_zero(tmp_path: Path) -> None:
    events = [
        event(1, "turn_start", turn=1),
        event(
            2,
            "tool_request",
            tool_call={
                "id": "write",
                "name": "write_file",
                "arguments": {"path": "a.py", "content": "SECRET-CONTENT"},
                "arguments_error": None,
            },
        ),
        event(3, "tool_result", tool_result={"content": "ok", "is_error": False}),
        event(4, "final", message="done"),
    ]

    code, output, server, client, _ = run_with_fakes(
        make_config(tmp_path),
        events,
    )

    rendered = "\n".join(output)
    assert code == 0
    assert "[THINK]" in rendered
    assert "[ACT] write_file a.py" in rendered
    assert "[OBS] ok" in rendered
    assert "[DONE] done" in rendered
    assert "SECRET-CONTENT" not in rendered
    assert client.sent == [("session-1", "task")]
    assert client.closed
    assert server.started and server.stopped


def test_error_event_returns_one_and_always_stops_server(tmp_path: Path) -> None:
    code, output, server, client, _ = run_with_fakes(
        make_config(tmp_path),
        [event(1, "error", message="failed")],
    )

    assert code == 1
    assert output == ["[ERROR] failed"]
    assert client.closed
    assert server.stopped


@pytest.mark.parametrize(
    ("answer", "expected"),
    [("y", True), ("yes", True), ("", False), ("no", False)],
)
def test_permission_answer_is_sent_to_server(
    answer: str,
    expected: bool,
    tmp_path: Path,
) -> None:
    events = [
        event(
            1,
            "tool_confirm",
            permission_id="permission-1",
            preview="a.py",
            tool_call={"name": "write_file", "arguments": {"path": "a.py"}},
        ),
        event(2, "final", message="done"),
    ]

    code, _, _, client, _ = run_with_fakes(
        make_config(tmp_path),
        events,
        input_func=lambda prompt: answer,
    )

    assert code == 0
    assert client.permissions == [("session-1", "permission-1", expected)]


def test_permission_eof_defaults_to_reject(tmp_path: Path) -> None:
    def eof(prompt: str) -> str:
        raise EOFError

    code, _, _, client, _ = run_with_fakes(
        make_config(tmp_path),
        [
            event(
                1,
                "tool_confirm",
                permission_id="permission-1",
                preview="pytest",
                tool_call={"name": "bash", "arguments": {"command": "pytest"}},
            ),
            event(2, "final", message="done"),
        ],
        input_func=eof,
    )

    assert code == 0
    assert client.permissions == [("session-1", "permission-1", False)]


def test_yes_sets_server_auto_approve_without_reading_input(tmp_path: Path) -> None:
    def forbidden_input(prompt: str) -> str:
        raise AssertionError("input must not be called")

    code, _, server, client, captured = run_with_fakes(
        make_config(tmp_path),
        [event(1, "final", message="done")],
        auto_approve=True,
        input_func=forbidden_input,
    )

    assert code == 0
    assert captured["auto_approve"] is True
    assert client.permissions == []
    assert server.stopped


def test_yes_never_approves_from_client_side(tmp_path: Path) -> None:
    def forbidden_input(prompt: str) -> str:
        raise AssertionError("input must not be called")

    code, output, _, client, _ = run_with_fakes(
        make_config(tmp_path),
        [
            event(
                1,
                "tool_confirm",
                permission_id="unexpected",
                preview="a.py",
                tool_call={"name": "write_file", "arguments": {"path": "a.py"}},
            )
        ],
        auto_approve=True,
        input_func=forbidden_input,
    )

    assert code == 1
    assert client.permissions == []
    assert any("不应产生 tool_confirm" in line for line in output)


def test_keyboard_interrupt_aborts_and_cleans_up(tmp_path: Path) -> None:
    class InterruptingClient(FakeHttpClient):
        def listen_events(self, session_id: str) -> Iterator[SseEvent]:
            raise KeyboardInterrupt
            yield

    config = make_config(tmp_path)
    output: list[str] = []
    client = InterruptingClient([])
    server = FakeServer(object())

    code = run_task(
        config,
        "task",
        output=output.append,
        app_factory=lambda *args, **kwargs: object(),
        server_factory=lambda app: server,
        http_client_factory=lambda url: client,
    )

    assert code == 130
    assert client.aborted == ["session-1"]
    assert client.closed
    assert server.stopped
    assert any("用户中止" in line for line in output)
