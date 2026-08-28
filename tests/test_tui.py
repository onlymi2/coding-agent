import asyncio
import subprocess
import sys
import threading
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from textual.widgets import Button, Input

from ke.client.http import HttpClientError, SseEvent
from ke.client.tui import KeTuiApp, PermissionModal, run_tui
from ke.config import ChannelConfig, KeConfig


def make_config(tmp_path: Path) -> KeConfig:
    return KeConfig(
        channel="deepseek",
        base_url="https://unused.example/v1",
        model="fake-model",
        api_key="test-key-must-not-render",
        workspace=tmp_path,
        channels={
            "deepseek": ChannelConfig("https://deepseek.example/v1", "d"),
            "openai": ChannelConfig("https://openai.example/v1", "o"),
            "local": ChannelConfig("http://localhost:8000/v1", "l"),
        },
    )


def event(event_id: int, name: str, **data: Any) -> SseEvent:
    return SseEvent(event_id, name, {"type": name, **data})


class FakeHttpClient:
    def __init__(
        self,
        batches: list[list[SseEvent]] | None = None,
        sessions: list[str] | None = None,
    ) -> None:
        self.batches = list(batches or [])
        self.sessions = list(sessions or ["session-2"])
        self.create_calls = 0
        self.sent: list[tuple[str, str]] = []
        self.listen_calls: list[tuple[str, int | None]] = []
        self.permissions: list[tuple[str, str, bool]] = []
        self.aborted: list[str] = []
        self.closed = False

    def create_session(self) -> str:
        self.create_calls += 1
        return self.sessions.pop(0)

    def send_message(self, session_id: str, content: str) -> None:
        self.sent.append((session_id, content))

    def listen_events(
        self,
        session_id: str,
        *,
        last_event_id: int | None = None,
        max_reconnects: int = 3,
    ) -> Iterator[SseEvent]:
        self.listen_calls.append((session_id, last_event_id))
        yield from self.batches.pop(0)

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


async def wait_until(
    pilot: Any,
    predicate: Callable[[], bool],
    message: str = "condition",
) -> None:
    for _ in range(150):
        if predicate():
            return
        await pilot.pause(0.01)
    raise AssertionError(f"timed out waiting for {message}")


async def submit(app: KeTuiApp, pilot: Any, value: str) -> None:
    task_input = app.query_one("#task-input", Input)
    task_input.value = value
    task_input.focus()
    await pilot.press("enter")
    await pilot.pause()


def test_tui_starts_with_status_and_creates_runtime_session(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    lifecycle: dict[str, object] = {}
    client = FakeHttpClient(sessions=["session-1"])

    class FakeServer:
        base_url = "http://127.0.0.1:40000"

        def start(self) -> None:
            lifecycle["started"] = True

        def stop(self) -> None:
            lifecycle["stopped"] = True

    class FakeTui:
        task_running = False

        def __init__(self, **kwargs: object) -> None:
            lifecycle.update(kwargs)

        def run(self) -> int:
            lifecycle["ran"] = True
            return 0

    code = run_tui(
        config,
        app_factory=lambda *args, **kwargs: object(),
        server_factory=lambda app: FakeServer(),
        http_client_factory=lambda url: client,
        tui_factory=FakeTui,
    )

    assert code == 0
    assert lifecycle["started"] is True
    assert lifecycle["ran"] is True
    assert lifecycle["stopped"] is True
    assert lifecycle["client"] is client
    assert lifecycle["session_id"] == "session-1"
    assert lifecycle["web_url"] == (
        "http://127.0.0.1:40000/?session=session-1"
    )
    assert client.create_calls == 1
    assert client.closed


def test_tui_layout_and_server_events_render_safely(tmp_path: Path) -> None:
    client = FakeHttpClient(
        batches=[
            [
                event(1, "turn_start", turn=1),
                event(
                    2,
                    "tool_request",
                    tool_call={
                        "id": "write",
                        "name": "write_file",
                        "arguments": {
                            "path": "a.py",
                            "content": "SECRET-WRITE-CONTENT",
                        },
                    },
                ),
                event(
                    3,
                    "tool_result",
                    tool_result={"content": "written", "is_error": False},
                ),
                event(4, "context_compact", message="history compacted"),
                event(5, "final", message="done"),
            ]
        ]
    )
    web_url = "http://127.0.0.1:40000/?session=session-1"
    app = KeTuiApp(
        make_config(tmp_path),
        client,
        "session-1",
        web_url=web_url,
    )

    async def scenario() -> None:
        async with app.run_test() as pilot:
            assert app.agent_status == "IDLE"
            assert "deepseek / fake-model · IDLE" in app._status_text()
            await submit(app, pilot, "  build file  ")
            await wait_until(pilot, lambda: not app.task_running, "final")

            rendered = "\n".join(app.event_lines)
            assert f"WEB {web_url}" in rendered
            assert client.sent == [("session-1", "build file")]
            assert "THINK turn 1" in rendered
            assert "ACT write_file a.py" in rendered
            assert "OBS written" in rendered
            assert "COMPACT history compacted" in rendered
            assert "DONE done" in rendered
            assert "SECRET-WRITE-CONTENT" not in rendered
            assert app.status_history[-5:] == [
                "THINK",
                "ACT",
                "OBSERVE",
                "COMPACT",
                "DONE",
            ]
            assert app.agent_status == "DONE"
            assert not app.query_one("#task-input", Input).disabled

    asyncio.run(scenario())


def test_tui_error_event_recovers_input(tmp_path: Path) -> None:
    client = FakeHttpClient(batches=[[event(1, "error", message="failed")]])
    app = KeTuiApp(make_config(tmp_path), client, "session-1")

    async def scenario() -> None:
        async with app.run_test() as pilot:
            await submit(app, pilot, "fail")
            await wait_until(pilot, lambda: not app.task_running, "error")
            assert app.agent_status == "ERROR"
            assert "ERROR failed" in app.event_lines
            assert not app.query_one("#task-input", Input).disabled

    asyncio.run(scenario())


class PermissionHttpClient(FakeHttpClient):
    def __init__(self) -> None:
        super().__init__()
        self.resolved = threading.Event()

    def listen_events(
        self,
        session_id: str,
        *,
        last_event_id: int | None = None,
        max_reconnects: int = 3,
    ) -> Iterator[SseEvent]:
        self.listen_calls.append((session_id, last_event_id))
        yield event(
            1,
            "tool_confirm",
            permission_id="permission-1",
            preview="pytest -q",
            tool_call={"name": "bash", "arguments": {"command": "pytest -q"}},
        )
        if self.resolved.wait(timeout=2):
            yield event(2, "final", message="done")

    def resolve_permission(
        self,
        session_id: str,
        permission_id: str,
        allow: bool,
    ) -> None:
        super().resolve_permission(session_id, permission_id, allow)
        self.resolved.set()


@pytest.mark.parametrize(
    ("choice", "allowed"),
    [("allow", True), ("deny", False), ("escape", False)],
)
def test_permission_modal_is_deny_by_default_and_resolves(
    choice: str,
    allowed: bool,
    tmp_path: Path,
) -> None:
    client = PermissionHttpClient()
    app = KeTuiApp(make_config(tmp_path), client, "session-1")

    async def scenario() -> None:
        async with app.run_test() as pilot:
            await submit(app, pilot, "run tests")
            await wait_until(
                pilot,
                lambda: (
                    isinstance(app.screen, PermissionModal)
                    and len(app.screen.query("#deny")) == 1
                ),
                "mounted permission modal",
            )
            modal = app.screen
            assert isinstance(modal, PermissionModal)
            assert modal.query_one("#deny", Button).has_focus
            if choice == "escape":
                await pilot.press("escape")
            else:
                await pilot.click(f"#{choice}")
            await wait_until(pilot, lambda: bool(client.permissions), "permission")
            await wait_until(pilot, lambda: not app.task_running, "permission final")

            assert client.permissions == [
                ("session-1", "permission-1", allowed)
            ]
            assert app.agent_status == "DONE"

    asyncio.run(scenario())


def test_help_channel_new_and_generation_guard(tmp_path: Path) -> None:
    client = FakeHttpClient(sessions=["session-2"])
    app = KeTuiApp(
        make_config(tmp_path),
        client,
        "session-1",
        web_url="http://127.0.0.1:40000/?session=session-1",
    )

    async def scenario() -> None:
        async with app.run_test() as pilot:
            app.last_event_id = 99
            await submit(app, pilot, "/help")
            await submit(app, pilot, "/channel")
            await submit(app, pilot, "/new")
            await wait_until(
                pilot,
                lambda: app.session_id == "session-2",
                "new session",
            )

            assert app.last_event_id is None
            assert app.web_url == (
                "http://127.0.0.1:40000/?session=session-2"
            )
            assert client.create_calls == 1
            rendered = "\n".join(app.event_lines)
            for command in ("/help", "/new", "/channel", "/abort", "/quit"):
                assert command in rendered
            for channel in ("deepseek", "openai", "local"):
                assert channel in rendered
            assert "test-key-must-not-render" not in rendered
            assert "已创建新会话" in rendered
            assert f"WEB {app.web_url}" in rendered

            before = list(app.event_lines)
            app._receive_event(
                event(100, "final", message="stale"),
                "session-1",
                0,
            )
            assert app.event_lines == before

    asyncio.run(scenario())


def test_second_task_uses_cursor_and_does_not_render_old_events(
    tmp_path: Path,
) -> None:
    first = [
        event(1, "turn_start", turn=1),
        event(2, "final", message="first"),
    ]
    second_with_history = [
        *first,
        event(3, "turn_start", turn=1),
        event(4, "final", message="second"),
    ]
    client = FakeHttpClient(batches=[first, second_with_history])
    app = KeTuiApp(make_config(tmp_path), client, "session-1")

    async def scenario() -> None:
        async with app.run_test() as pilot:
            await submit(app, pilot, "task A")
            await wait_until(pilot, lambda: not app.task_running, "task A")
            await submit(app, pilot, "task B")
            await wait_until(pilot, lambda: not app.task_running, "task B")

            assert client.listen_calls == [
                ("session-1", None),
                ("session-1", 2),
            ]
            assert app.event_lines.count("DONE first") == 1
            assert app.event_lines.count("DONE second") == 1
            assert app.event_lines.count("THINK turn 1") == 2
            assert app.last_event_id == 4

    asyncio.run(scenario())


class BlockingHttpClient(FakeHttpClient):
    def __init__(self) -> None:
        super().__init__()
        self.listening = threading.Event()
        self.release = threading.Event()

    def listen_events(
        self,
        session_id: str,
        *,
        last_event_id: int | None = None,
        max_reconnects: int = 3,
    ) -> Iterator[SseEvent]:
        self.listen_calls.append((session_id, last_event_id))
        self.listening.set()
        if self.release.wait(timeout=2):
            yield event(1, "error", message="aborted")

    def abort(self, session_id: str) -> None:
        super().abort(session_id)
        self.release.set()


def test_blocking_sse_does_not_block_help_or_abort(tmp_path: Path) -> None:
    client = BlockingHttpClient()
    app = KeTuiApp(make_config(tmp_path), client, "session-1")

    async def scenario() -> None:
        async with app.run_test() as pilot:
            await submit(app, pilot, "long task")
            await wait_until(pilot, client.listening.is_set, "listener start")
            await submit(app, pilot, "/help")
            assert any(line.startswith("/help") for line in app.event_lines)
            await submit(app, pilot, "/abort")
            await wait_until(pilot, lambda: bool(client.aborted), "abort")
            await wait_until(pilot, lambda: not app.task_running, "abort event")

            assert client.aborted == ["session-1"]
            assert app.agent_status == "ERROR"

    asyncio.run(scenario())


@pytest.mark.parametrize("quit_action", ["command", "action"])
def test_quit_only_requests_textual_exit_and_ctrl_c_uses_same_action(
    quit_action: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeHttpClient()
    app = KeTuiApp(make_config(tmp_path), client, "session-1")
    app.task_running = True
    exit_calls: list[int] = []
    monkeypatch.setattr(
        app,
        "exit",
        lambda result=None, **kwargs: exit_calls.append(
            0 if result is None else int(result)
        ),
    )

    if quit_action == "command":
        app._handle_command("/quit")
    else:
        app.action_safe_quit()

    assert client.aborted == []
    assert exit_calls == [0]
    assert app._closing
    assert any(
        binding.key == "ctrl+c" and binding.action == "safe_quit"
        for binding in app.BINDINGS
    )


def test_late_permission_callback_does_not_regress_observe_status(
    tmp_path: Path,
) -> None:
    client = FakeHttpClient()
    app = KeTuiApp(make_config(tmp_path), client, "session-1")
    app.task_running = True
    app._pending_permission_id = "permission-1"
    app.agent_status = "OBSERVE"
    logged: list[str] = []
    app._write = logged.append  # type: ignore[method-assign]

    app._permission_resolved(
        "session-1",
        0,
        "permission-1",
        True,
    )

    assert app._pending_permission_id is None
    assert app.agent_status == "OBSERVE"
    assert logged == ["权限：Allow"]


def test_permission_409_is_handled_without_aborting_agent(
    tmp_path: Path,
) -> None:
    class ConflictClient(FakeHttpClient):
        def resolve_permission(
            self,
            session_id: str,
            permission_id: str,
            allow: bool,
        ) -> None:
            raise HttpClientError(
                "HTTP 409：permission 已经完成",
                status_code=409,
            )

    client = ConflictClient()
    app = KeTuiApp(make_config(tmp_path), client, "session-1")
    app.task_running = True
    app._pending_permission_id = "permission-1"
    app.agent_status = "CONFIRM"
    logged: list[str] = []
    app._write = logged.append  # type: ignore[method-assign]
    app._set_status = (  # type: ignore[method-assign]
        lambda value: setattr(app, "agent_status", value)
    )
    app._call_ui = (  # type: ignore[method-assign]
        lambda callback, *args: callback(*args)
    )

    app._permission_worker(
        "session-1",
        0,
        "permission-1",
        True,
    )

    assert client.aborted == []
    assert app._pending_permission_id is None
    assert app.agent_status == "ACT"
    assert logged == ["权限已由其他客户端处理"]
    assert app.task_running


def test_tool_result_closes_permission_modal_without_second_post(
    tmp_path: Path,
) -> None:
    client = FakeHttpClient()
    app = KeTuiApp(make_config(tmp_path), client, "session-1")

    async def scenario() -> None:
        async with app.run_test() as pilot:
            app.task_running = True
            app._receive_event(
                event(
                    1,
                    "tool_confirm",
                    permission_id="permission-1",
                    preview="pytest -q",
                    tool_call={"name": "bash", "arguments": {}},
                ),
                "session-1",
                0,
            )
            await wait_until(
                pilot,
                lambda: (
                    isinstance(app.screen, PermissionModal)
                    and len(app.screen.query("#deny")) == 1
                ),
                "mounted permission modal",
            )

            app._receive_event(
                event(
                    2,
                    "tool_result",
                    tool_result={"content": "done", "is_error": False},
                ),
                "session-1",
                0,
            )
            await wait_until(
                pilot,
                lambda: not isinstance(app.screen, PermissionModal),
                "permission modal close",
            )

            assert app._pending_permission_id is None
            assert client.permissions == []
            assert app.agent_status == "OBSERVE"
            assert "OBS done" in app.event_lines

    asyncio.run(scenario())


def test_run_tui_aborts_active_task_and_always_cleans_resources(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    client = FakeHttpClient(sessions=["session-1"])

    class FakeServer:
        base_url = "http://127.0.0.1:40000"
        stopped = False

        def start(self) -> None:
            pass

        def stop(self) -> None:
            self.stopped = True

    class ActiveTui:
        task_running = True

        def __init__(self, **kwargs: object) -> None:
            pass

        def run(self) -> int:
            return 0

    server = FakeServer()
    code = run_tui(
        config,
        app_factory=lambda *args, **kwargs: object(),
        server_factory=lambda app: server,
        http_client_factory=lambda url: client,
        tui_factory=ActiveTui,
    )

    assert code == 0
    assert client.aborted == ["session-1"]
    assert client.closed
    assert server.stopped


@pytest.mark.parametrize(
    ("failure", "expected_code"),
    [(KeyboardInterrupt(), 130), (RuntimeError("boom"), 1)],
)
def test_run_tui_interrupt_or_error_still_cleans_resources(
    failure: BaseException,
    expected_code: int,
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    client = FakeHttpClient(sessions=["session-1"])

    class FakeServer:
        base_url = "http://127.0.0.1:40000"
        stopped = False

        def start(self) -> None:
            pass

        def stop(self) -> None:
            self.stopped = True

    class FailingTui:
        task_running = True

        def __init__(self, **kwargs: object) -> None:
            pass

        def run(self) -> int:
            raise failure

    server = FakeServer()
    errors: list[str] = []
    code = run_tui(
        config,
        app_factory=lambda *args, **kwargs: object(),
        server_factory=lambda app: server,
        http_client_factory=lambda url: client,
        tui_factory=FailingTui,
        error_output=errors.append,
    )

    assert code == expected_code
    assert client.aborted == ["session-1"]
    assert client.closed
    assert server.stopped
    assert "test-key-must-not-render" not in "\n".join(errors)


def test_import_tui_has_no_runtime_side_effects() -> None:
    result = subprocess.run(
        [sys.executable, "-c", "import ke.client.tui; print('ok')"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert result.stdout.strip() == "ok"
