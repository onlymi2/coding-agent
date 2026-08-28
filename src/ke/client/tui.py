import sys
from collections.abc import Callable, Iterator
from typing import Any, Protocol

from starlette.applications import Starlette
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Footer, Input, RichLog, Static

from ke.client.http import KeHttpClient, SseEvent
from ke.client.run import ServerLike, safe_tool_summary
from ke.config import KeConfig
from ke.server.runtime import EmbeddedServer, build_app


HELP_LINES = (
    "/help     显示命令说明",
    "/new      创建新会话",
    "/channel  列出已知模型渠道",
    "/abort    中止当前任务",
    "/quit     退出 ke",
)


class HttpClientLike(Protocol):
    def create_session(self) -> str: ...

    def send_message(self, session_id: str, content: str) -> None: ...

    def listen_events(
        self,
        session_id: str,
        *,
        last_event_id: int | None = None,
        max_reconnects: int = 3,
    ) -> Iterator[SseEvent]: ...

    def resolve_permission(
        self,
        session_id: str,
        permission_id: str,
        allow: bool,
    ) -> None: ...

    def abort(self, session_id: str) -> None: ...

    def close(self) -> None: ...


class PermissionModal(ModalScreen[bool]):
    """A deny-by-default confirmation screen for one dangerous tool call."""

    BINDINGS = [Binding("escape", "deny", "Deny", show=False)]
    CSS = """
    PermissionModal {
        align: center middle;
    }
    #permission-dialog {
        width: 64;
        height: auto;
        padding: 1 2;
        border: round $accent;
        background: $surface;
    }
    #permission-buttons {
        height: auto;
        align-horizontal: right;
        margin-top: 1;
    }
    #permission-buttons Button {
        margin-left: 1;
    }
    """

    def __init__(self, tool_name: str, preview: str) -> None:
        super().__init__()
        self.tool_name = tool_name
        self.preview = preview[:200]

    def compose(self) -> ComposeResult:
        with Vertical(id="permission-dialog"):
            yield Static("确认执行本地工具", markup=False)
            yield Static(f"工具：{self.tool_name}", markup=False)
            yield Static(f"预览：{self.preview}", markup=False)
            with Horizontal(id="permission-buttons"):
                yield Button("Deny", id="deny", variant="error")
                yield Button("Allow", id="allow", variant="success")

    def on_mount(self) -> None:
        self.query_one("#deny", Button).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss(event.button.id == "allow")

    def action_deny(self) -> None:
        self.dismiss(False)


class KeTuiApp(App[int]):
    """Textual presentation layer backed only by KeHttpClient calls."""

    CSS = """
    Screen {
        layout: vertical;
    }
    #status-bar {
        height: 3;
        padding: 1 2;
        background: $boost;
        text-style: bold;
    }
    #event-log {
        height: 1fr;
        border: round $accent;
        padding: 0 1;
    }
    #task-input {
        dock: bottom;
        margin: 0 1;
    }
    Footer {
        height: 1;
    }
    """
    BINDINGS = [
        Binding("ctrl+c", "safe_quit", "退出", priority=True),
    ]

    def __init__(
        self,
        config: KeConfig,
        client: HttpClientLike,
        session_id: str,
    ) -> None:
        super().__init__()
        self.config = config
        self.client = client
        self.session_id = session_id
        self.last_event_id: int | None = None
        self.agent_status = "IDLE"
        self.status_history = ["IDLE"]
        self.task_running = False
        self.event_lines: list[str] = []
        self._generation = 0
        self._creating_session = False
        self._pending_permission_id: str | None = None
        self._closing = False

    def compose(self) -> ComposeResult:
        yield Static(self._status_text(), id="status-bar", markup=False)
        yield RichLog(id="event-log", wrap=True, markup=False, highlight=False)
        yield Input(
            placeholder="输入任务，/help 查看命令",
            id="task-input",
        )
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#task-input", Input).focus()

    def _status_text(self) -> str:
        return (
            f"ke · {self.config.channel} / {self.config.model} · "
            f"{self.agent_status}"
        )

    def _set_status(self, value: str) -> None:
        self.agent_status = value
        self.status_history.append(value)
        self.query_one("#status-bar", Static).update(self._status_text())

    def _write(self, line: str) -> None:
        self.event_lines.append(line)
        self.query_one("#event-log", RichLog).write(line)

    def _set_input_enabled(self, enabled: bool) -> None:
        task_input = self.query_one("#task-input", Input)
        task_input.disabled = not enabled
        if enabled:
            task_input.focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        value = event.value.strip()
        event.input.value = ""
        if not value:
            return
        if value.startswith("/"):
            self._handle_command(value)
            return
        if self.task_running or self._creating_session:
            self._write("ERROR 当前任务仍在运行")
            return

        self._write(f"你: {value}")
        self.task_running = True
        self._set_status("THINK")
        session_id = self.session_id
        generation = self._generation
        cursor = self.last_event_id
        self.run_worker(
            lambda: self._task_worker(value, session_id, generation, cursor),
            name=f"task-{generation}",
            group=f"task-{generation}",
            thread=True,
            exit_on_error=False,
        )

    def _call_ui(self, callback: Callable[..., Any], *args: object) -> None:
        if self._closing:
            return
        try:
            self.call_from_thread(callback, *args)
        except RuntimeError:
            pass

    def _task_worker(
        self,
        task: str,
        session_id: str,
        generation: int,
        cursor: int | None,
    ) -> None:
        try:
            self.client.send_message(session_id, task)
            for event in self.client.listen_events(
                session_id,
                last_event_id=cursor,
            ):
                if cursor is not None and event.id <= cursor:
                    continue
                cursor = event.id
                self._call_ui(
                    self._receive_event,
                    event,
                    session_id,
                    generation,
                )
                if event.event in {"final", "error"}:
                    return
        except Exception as exc:
            self._call_ui(
                self._task_failed,
                session_id,
                generation,
                type(exc).__name__,
            )

    def _receive_event(
        self,
        event: SseEvent,
        session_id: str,
        generation: int,
    ) -> None:
        if (
            self._closing
            or session_id != self.session_id
            or generation != self._generation
        ):
            return
        if self.last_event_id is not None and event.id <= self.last_event_id:
            return
        self.last_event_id = event.id

        if event.event == "turn_start":
            self._set_status("THINK")
            self._write(f"THINK turn {event.data.get('turn', '')}")
        elif event.event == "tool_request":
            self._set_status("ACT")
            self._write(f"ACT {safe_tool_summary(event.data)}")
        elif event.event == "tool_result":
            self._set_status("OBSERVE")
            result = event.data.get("tool_result")
            content = result.get("content", "") if isinstance(result, dict) else ""
            self._write(f"OBS {content}")
        elif event.event in {"context_compact", "context_summary"}:
            self._set_status("COMPACT")
            self._write(f"COMPACT {self._event_message(event)}")
        elif event.event == "tool_confirm":
            self._show_permission(event)
        elif event.event == "final":
            self._set_status("DONE")
            self._write(f"DONE {self._event_message(event)}")
            self._finish_task()
        elif event.event == "error":
            self._set_status("ERROR")
            self._write(f"ERROR {self._event_message(event)}")
            self._finish_task()

    @staticmethod
    def _event_message(event: SseEvent) -> str:
        message = event.data.get("message", "")
        return message if isinstance(message, str) else str(message)

    def _finish_task(self) -> None:
        self.task_running = False
        self._pending_permission_id = None
        self._set_input_enabled(True)

    def _task_failed(
        self,
        session_id: str,
        generation: int,
        error_type: str,
    ) -> None:
        if session_id != self.session_id or generation != self._generation:
            return
        self._set_status("ERROR")
        self._write(f"ERROR HTTP/SSE 失败（{error_type}）")
        self._finish_task()

    def _show_permission(self, event: SseEvent) -> None:
        permission_id = event.data.get("permission_id")
        call = event.data.get("tool_call")
        if not isinstance(permission_id, str) or not permission_id:
            self._task_failed(
                self.session_id,
                self._generation,
                "InvalidPermission",
            )
            return
        if self._pending_permission_id is not None:
            return
        tool_name = call.get("name", "tool") if isinstance(call, dict) else "tool"
        preview = str(event.data.get("preview", ""))[:200]
        self._pending_permission_id = permission_id
        self._set_status("CONFIRM")
        self.push_screen(
            PermissionModal(str(tool_name), preview),
            self._permission_decided,
        )

    def _permission_decided(self, allowed: bool | None) -> None:
        permission_id = self._pending_permission_id
        if permission_id is None:
            return
        session_id = self.session_id
        generation = self._generation
        self.run_worker(
            lambda: self._permission_worker(
                session_id,
                generation,
                permission_id,
                allowed is True,
            ),
            name=f"permission-{permission_id}",
            group="permission",
            thread=True,
            exit_on_error=False,
            exclusive=True,
        )

    def _permission_worker(
        self,
        session_id: str,
        generation: int,
        permission_id: str,
        allowed: bool,
    ) -> None:
        try:
            self.client.resolve_permission(
                session_id,
                permission_id,
                allowed,
            )
        except Exception as exc:
            try:
                self.client.abort(session_id)
            except Exception:
                pass
            self._call_ui(
                self._permission_failed,
                session_id,
                generation,
                type(exc).__name__,
            )
            return
        self._call_ui(
            self._permission_resolved,
            session_id,
            generation,
            permission_id,
            allowed,
        )

    def _permission_resolved(
        self,
        session_id: str,
        generation: int,
        permission_id: str,
        allowed: bool,
    ) -> None:
        if session_id != self.session_id or generation != self._generation:
            return
        if permission_id != self._pending_permission_id:
            return
        self._pending_permission_id = None
        if not self.task_running:
            return
        if self.agent_status == "CONFIRM":
            self._set_status("ACT")
        self._write("权限：Allow" if allowed else "权限：Deny")

    def _permission_failed(
        self,
        session_id: str,
        generation: int,
        error_type: str,
    ) -> None:
        if session_id != self.session_id or generation != self._generation:
            return
        self._pending_permission_id = None
        self._set_status("ERROR")
        self._write(f"ERROR 权限提交失败（{error_type}），已请求中止")
        self._finish_task()

    def _handle_command(self, command: str) -> None:
        if command == "/help":
            for line in HELP_LINES:
                self._write(line)
        elif command == "/channel":
            names = sorted(set(self.config.channels) | {self.config.channel})
            for name in names:
                marker = "*" if name == self.config.channel else "-"
                self._write(f"CHANNEL {marker} {name}")
        elif command == "/abort":
            self._request_abort()
        elif command == "/new":
            self._request_new_session()
        elif command == "/quit":
            self.action_safe_quit()
        else:
            self._write("ERROR 未知命令；使用 /help 查看支持的命令")

    def _request_abort(self) -> None:
        if not self.task_running:
            self._write("当前没有运行中的任务")
            return
        self._set_status("ABORTING")
        session_id = self.session_id
        generation = self._generation
        self.run_worker(
            lambda: self._abort_worker(session_id, generation),
            name=f"abort-{generation}",
            group="control",
            thread=True,
            exit_on_error=False,
        )

    def _abort_worker(self, session_id: str, generation: int) -> None:
        try:
            self.client.abort(session_id)
        except Exception as exc:
            self._call_ui(
                self._control_failed,
                session_id,
                generation,
                "abort",
                type(exc).__name__,
            )

    def _request_new_session(self) -> None:
        if self.task_running:
            self._write("ERROR 当前任务运行中，请先 /abort")
            return
        if self._creating_session:
            return
        self._creating_session = True
        self._set_input_enabled(False)
        next_generation = self._generation + 1
        self.run_worker(
            lambda: self._new_session_worker(next_generation),
            name=f"new-session-{next_generation}",
            group="control",
            thread=True,
            exit_on_error=False,
            exclusive=True,
        )

    def _new_session_worker(self, generation: int) -> None:
        try:
            session_id = self.client.create_session()
        except Exception as exc:
            self._call_ui(
                self._new_session_failed,
                generation,
                type(exc).__name__,
            )
            return
        self._call_ui(self._new_session_ready, session_id, generation)

    def _new_session_ready(self, session_id: str, generation: int) -> None:
        if self._closing:
            return
        self.session_id = session_id
        self._generation = generation
        self.last_event_id = None
        self._creating_session = False
        self._set_status("IDLE")
        self._write("已创建新会话")
        self._set_input_enabled(True)

    def _new_session_failed(self, generation: int, error_type: str) -> None:
        if generation != self._generation + 1:
            return
        self._creating_session = False
        self._set_status("ERROR")
        self._write(f"ERROR 创建会话失败（{error_type}）")
        self._set_input_enabled(True)

    def _control_failed(
        self,
        session_id: str,
        generation: int,
        operation: str,
        error_type: str,
    ) -> None:
        if session_id != self.session_id or generation != self._generation:
            return
        self._set_status("ERROR")
        self._write(f"ERROR {operation} 失败（{error_type}）")

    def action_safe_quit(self) -> None:
        if self._closing:
            return
        self._closing = True
        self.exit(0)


def run_tui(
    config: KeConfig,
    *,
    app_factory: Callable[..., Starlette] = build_app,
    server_factory: Callable[[Starlette], ServerLike] = EmbeddedServer,
    http_client_factory: Callable[[str], HttpClientLike] = KeHttpClient,
    tui_factory: Callable[..., KeTuiApp] = KeTuiApp,
    error_output: Callable[[str], None] | None = None,
) -> int:
    """Start the embedded Server and attach the Textual HTTP/SSE client."""

    emit_error = error_output or (lambda text: print(text, file=sys.stderr))
    server: ServerLike | None = None
    client: HttpClientLike | None = None
    tui: KeTuiApp | None = None
    session_id: str | None = None
    exit_code = 1
    cleanup_failed = False
    try:
        server_app = app_factory(config, auto_approve=False)
        server = server_factory(server_app)
        server.start()
        client = http_client_factory(server.base_url)
        session_id = client.create_session()
        tui = tui_factory(config=config, client=client, session_id=session_id)
        result = tui.run()
        exit_code = 0 if result is None else int(result)
    except KeyboardInterrupt:
        exit_code = 130
    except Exception as exc:
        emit_error(f"TUI 启动或运行失败（{type(exc).__name__}）")
        exit_code = 1
    finally:
        if (
            client is not None
            and session_id is not None
            and tui is not None
            and tui.task_running
        ):
            try:
                client.abort(session_id)
            except Exception:
                pass
        if client is not None:
            try:
                client.close()
            except Exception:
                cleanup_failed = True
        if server is not None:
            try:
                server.stop()
            except Exception:
                cleanup_failed = True
        if cleanup_failed:
            emit_error("TUI 本地运行资源清理失败")

    return 1 if cleanup_failed and exit_code != 130 else exit_code
