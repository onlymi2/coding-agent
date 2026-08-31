import json
import threading
import uuid
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from importlib.resources import files
from typing import Any

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, StreamingResponse
from starlette.routing import Route

from ke.agent.context import AgentContext
from ke.agent.events import AgentEvent
from ke.agent.loop import AgentState, run_agent
from ke.agent.prompts import build_system_prompt
from ke.config import KeConfig
from ke.llm.protocol import LlmClient
from ke.llm.types import Message, ToolCall
from ke.safety.confirm import (
    PermissionAlreadyResolvedError,
    PermissionGate,
    UnknownPermissionError,
)
from ke.safety.sandbox import WorkspaceSandbox
from ke.tools.registry import ToolRegistry
from ke.tools.types import ToolResult


LlmFactory = Callable[[], LlmClient]
RegistryFactory = Callable[[WorkspaceSandbox], ToolRegistry]


def load_static_html() -> str:
    """Load the packaged Web client without initializing runtime services."""

    return files("ke.server").joinpath("static.html").read_text(encoding="utf-8")


def _tool_call_json(call: ToolCall) -> dict[str, Any]:
    return {
        "id": call.id,
        "name": call.name,
        "arguments": call.arguments,
        "arguments_error": call.arguments_error,
    }


def _tool_result_json(result: ToolResult) -> dict[str, Any]:
    return {
        "content": result.content,
        "is_error": result.is_error,
    }


def event_to_json(event: AgentEvent) -> dict[str, Any]:
    """Serialize an AgentEvent without depending on dataclass internals."""

    payload: dict[str, Any] = {
        "type": event.type,
        "turn": event.turn,
    }
    if event.message is not None:
        payload["message"] = event.message
    if event.tool_call is not None:
        payload["tool_call"] = _tool_call_json(event.tool_call)
    if event.tool_result is not None:
        payload["tool_result"] = _tool_result_json(event.tool_result)
    if event.permission_id is not None:
        payload["permission_id"] = event.permission_id
    if event.preview is not None:
        payload["preview"] = event.preview
    return payload


@dataclass(frozen=True)
class StoredEvent:
    event_id: int
    event_type: str
    data: dict[str, Any]


class EventHistory:
    """Append-only broadcast history with a cursor per subscriber."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._events: list[StoredEvent] = []
        self._closed = False

    def publish(self, event: AgentEvent) -> StoredEvent:
        with self._condition:
            stored = StoredEvent(
                event_id=len(self._events) + 1,
                event_type=event.type,
                data=event_to_json(event),
            )
            self._events.append(stored)
            self._condition.notify_all()
            return stored

    def snapshot(self, after_id: int = 0) -> list[StoredEvent]:
        with self._condition:
            return list(self._events[max(0, after_id) :])

    def wait_after(
        self,
        after_id: int,
        timeout: float = 15.0,
    ) -> list[StoredEvent]:
        with self._condition:
            self._condition.wait_for(
                lambda: len(self._events) > after_id or self._closed,
                timeout=timeout,
            )
            return list(self._events[max(0, after_id) :])

    @property
    def closed(self) -> bool:
        with self._condition:
            return self._closed

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()

    def wake(self) -> None:
        with self._condition:
            self._condition.notify_all()


class AgentSession:
    def __init__(
        self,
        *,
        session_id: str,
        state: AgentState,
        llm: LlmClient,
        summary_llm: LlmClient | None,
        tools: ToolRegistry,
        permissions: PermissionGate,
    ) -> None:
        self.id = session_id
        self.state = state
        self.llm = llm
        self.summary_llm = summary_llm
        self.tools = tools
        self.permissions = permissions
        self.events = EventHistory()
        self._condition = threading.Condition()
        self._running = False
        self._worker: threading.Thread | None = None

    @property
    def running(self) -> bool:
        with self._condition:
            return self._running

    @property
    def worker(self) -> threading.Thread | None:
        with self._condition:
            return self._worker

    def start(self, task: str) -> bool:
        with self._condition:
            if self._running:
                return False
            self.permissions.reset()
            self.state.turn = 0
            self.state.cancelled = False
            self._running = True
            worker = threading.Thread(
                target=self._run,
                args=(task,),
                name=f"ke-session-{self.id}",
                daemon=True,
            )
            self._worker = worker
            worker.start()
            return True

    def _run(self, task: str) -> None:
        try:
            for event in run_agent(
                task,
                self.llm,
                self.tools,
                self.state,
                summary_llm=self.summary_llm,
                confirmation=self.permissions,
            ):
                if (
                    event.type == "error"
                    and event.message is not None
                    and event.message.startswith("LLM 调用失败：")
                ):
                    event = AgentEvent(
                        type="error",
                        turn=event.turn,
                        message="LLM 调用失败",
                    )
                self.events.publish(event)
        except Exception as exc:
            self.events.publish(
                AgentEvent(
                    type="error",
                    turn=self.state.turn,
                    message=f"Agent worker异常（{type(exc).__name__}）",
                )
            )
        finally:
            self.permissions.cancel_all()
            with self._condition:
                self._running = False
                self._condition.notify_all()
            self.events.wake()

    def abort(self) -> None:
        with self._condition:
            self.state.abort()
            self.permissions.cancel_all()
        self.events.wake()

    def wait_until_idle(self, timeout: float = 5.0) -> bool:
        with self._condition:
            return self._condition.wait_for(lambda: not self._running, timeout)


class SessionRuntime:
    def __init__(
        self,
        config: KeConfig,
        llm_factory: LlmFactory,
        summary_llm_factory: LlmFactory | None,
        registry_factory: RegistryFactory | None,
        auto_approve: bool,
    ) -> None:
        self.config = config
        self.llm_factory = llm_factory
        self.summary_llm_factory = summary_llm_factory
        self.registry_factory = registry_factory
        self.auto_approve = auto_approve
        self.workspace = WorkspaceSandbox(config.workspace).workspace
        self._lock = threading.Lock()
        self._sessions: dict[str, AgentSession] = {}
        self._closing = False

    def create_session(self) -> AgentSession:
        with self._lock:
            if self._closing:
                raise RuntimeError("runtime正在关闭")
        sandbox = WorkspaceSandbox(self.workspace)
        registry = (
            self.registry_factory(sandbox)
            if self.registry_factory is not None
            else ToolRegistry(
                sandbox,
                max_output_chars=self.config.max_tool_output_chars,
            )
        )
        context = AgentContext(
            messages=[
                Message(
                    role="system",
                    content=build_system_prompt(self.workspace),
                )
            ],
            compact_threshold_tokens=self.config.compact_threshold_tokens,
            max_tool_output_chars=self.config.max_tool_output_chars,
        )
        session = AgentSession(
            session_id=str(uuid.uuid4()),
            state=AgentState(context=context, max_turns=self.config.max_turns),
            llm=self.llm_factory(),
            summary_llm=(
                self.summary_llm_factory()
                if self.summary_llm_factory is not None
                else None
            ),
            tools=registry,
            permissions=PermissionGate(auto_approve=self.auto_approve),
        )
        with self._lock:
            if self._closing:
                session.events.close()
                raise RuntimeError("runtime正在关闭")
            self._sessions[session.id] = session
        return session

    def get(self, session_id: str) -> AgentSession | None:
        with self._lock:
            return self._sessions.get(session_id)

    def shutdown(self) -> None:
        """Stop active work and wake every session-level SSE subscriber."""

        with self._lock:
            if self._closing:
                return
            self._closing = True
            sessions = list(self._sessions.values())
        for session in sessions:
            session.abort()
            session.events.close()


def _error(message: str, status_code: int) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status_code)


async def _json_body(request: Request) -> dict[str, Any] | None:
    try:
        value = await request.json()
    except Exception:
        return None
    return value if isinstance(value, dict) else None


def _sse_frame(event: StoredEvent) -> str:
    data = json.dumps(event.data, ensure_ascii=False, separators=(",", ":"))
    return f"id: {event.event_id}\nevent: {event.event_type}\ndata: {data}\n\n"


def _event_stream(session: AgentSession, after_id: int) -> Iterator[str]:
    cursor = max(0, after_id)
    while True:
        pending = session.events.wait_after(cursor)
        for event in pending:
            yield _sse_frame(event)
            cursor = event.event_id
        if session.events.closed:
            return
        if not pending:
            yield ": keep-alive\n\n"


def create_app(
    config: KeConfig,
    *,
    llm_factory: LlmFactory,
    summary_llm_factory: LlmFactory | None = None,
    registry_factory: RegistryFactory | None = None,
    auto_approve: bool = False,
) -> Starlette:
    """Create six business APIs plus the packaged Web page."""

    runtime = SessionRuntime(
        config,
        llm_factory,
        summary_llm_factory,
        registry_factory,
        auto_approve,
    )

    @asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncIterator[None]:
        try:
            yield
        finally:
            runtime.shutdown()

    async def health(request: Request) -> JSONResponse:
        return JSONResponse({"status": "ok"})

    async def web_page(request: Request) -> HTMLResponse:
        return HTMLResponse(load_static_html())

    async def create_session(request: Request) -> JSONResponse:
        try:
            session = runtime.create_session()
        except Exception:
            return _error("session 初始化失败", 500)
        return JSONResponse({"id": session.id}, status_code=201)

    async def post_message(request: Request) -> JSONResponse:
        session = runtime.get(request.path_params["id"])
        if session is None:
            return _error("session 不存在", 404)
        body = await _json_body(request)
        content = body.get("content") if body is not None else None
        if not isinstance(content, str) or not content.strip():
            return _error("content 必须是非空字符串", 400)
        if not session.start(content.strip()):
            return _error("session 正在运行", 409)
        return JSONResponse({"status": "accepted"}, status_code=202)

    async def get_events(request: Request) -> JSONResponse | StreamingResponse:
        session = runtime.get(request.path_params["id"])
        if session is None:
            return _error("session 不存在", 404)
        raw_last_id = request.headers.get("last-event-id", "0")
        try:
            after_id = max(0, int(raw_last_id))
        except ValueError:
            after_id = 0
        return StreamingResponse(
            _event_stream(session, after_id),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )

    async def post_permission(request: Request) -> JSONResponse:
        session = runtime.get(request.path_params["id"])
        if session is None:
            return _error("session 不存在", 404)
        body = await _json_body(request)
        allow = body.get("allow") if body is not None else None
        if not isinstance(allow, bool):
            return _error("allow 必须是 boolean", 400)
        permission_id = request.path_params["pid"]
        try:
            permission = session.permissions.resolve(permission_id, allow)
        except UnknownPermissionError:
            return _error("permission 不存在", 404)
        except PermissionAlreadyResolvedError:
            return _error("permission 已经完成", 409)
        return JSONResponse(
            {
                "permission_id": permission.permission_id,
                "allowed": allow,
            }
        )

    async def abort(request: Request) -> JSONResponse:
        session = runtime.get(request.path_params["id"])
        if session is None:
            return _error("session 不存在", 404)
        session.abort()
        return JSONResponse({"status": "aborting"}, status_code=202)

    app = Starlette(
        routes=[
            Route("/", web_page, methods=["GET"]),
            Route("/health", health, methods=["GET"]),
            Route("/session", create_session, methods=["POST"]),
            Route("/session/{id}/message", post_message, methods=["POST"]),
            Route("/session/{id}/events", get_events, methods=["GET"]),
            Route(
                "/session/{id}/permissions/{pid}",
                post_permission,
                methods=["POST"],
            ),
            Route("/session/{id}/abort", abort, methods=["POST"]),
        ],
        lifespan=lifespan,
    )
    app.state.runtime = runtime
    return app
