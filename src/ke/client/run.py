from collections.abc import Callable
from typing import Any, Protocol

from starlette.applications import Starlette

from ke.client.http import HttpClientError, KeHttpClient, SseEvent
from ke.config import KeConfig
from ke.server.runtime import EmbeddedServer, EmbeddedServerError, build_app


class ServerLike(Protocol):
    @property
    def base_url(self) -> str: ...

    def start(self) -> None: ...

    def stop(self) -> None: ...


def safe_tool_summary(data: dict[str, Any]) -> str:
    call = data.get("tool_call")
    if not isinstance(call, dict):
        return "unknown"
    name = str(call.get("name") or "unknown")
    arguments = call.get("arguments")
    if not isinstance(arguments, dict):
        return name
    if name in {"read_file", "write_file", "edit_file", "list_dir"}:
        detail = arguments.get("path", "")
    elif name == "bash":
        detail = str(arguments.get("command", ""))[:200]
    elif name == "grep":
        detail = arguments.get("pattern", "")
    else:
        detail = ""
    return f"{name} {detail}".rstrip()


def _message(data: dict[str, Any]) -> str:
    value = data.get("message", "")
    return value if isinstance(value, str) else str(value)


def _print_event(event: SseEvent, output: Callable[[str], None]) -> None:
    if event.event == "turn_start":
        output(f"[THINK] turn {event.data.get('turn', '')}")
    elif event.event == "tool_request":
        output(f"[ACT] {safe_tool_summary(event.data)}")
    elif event.event == "tool_result":
        result = event.data.get("tool_result")
        content = result.get("content", "") if isinstance(result, dict) else ""
        output(f"[OBS] {content}")
    elif event.event in {"context_compact", "context_summary"}:
        output(f"[COMPACT] {_message(event.data)}")
    elif event.event == "final":
        output(f"[DONE] {_message(event.data)}")
    elif event.event == "error":
        output(f"[ERROR] {_message(event.data)}")


def _confirm(
    event: SseEvent,
    input_func: Callable[[str], str],
) -> bool:
    call = event.data.get("tool_call")
    name = call.get("name", "tool") if isinstance(call, dict) else "tool"
    preview = event.data.get("preview", "")
    prompt = f"Allow {name} {preview}? [y/N]: ".replace("  ", " ")
    try:
        answer = input_func(prompt)
    except Exception:
        return False
    return answer.strip().casefold() in {"y", "yes"}


def run_task(
    config: KeConfig,
    task: str,
    *,
    auto_approve: bool = False,
    output: Callable[[str], None] = print,
    input_func: Callable[[str], str] = input,
    app_factory: Callable[..., Starlette] = build_app,
    server_factory: Callable[[Starlette], ServerLike] = EmbeddedServer,
    http_client_factory: Callable[[str], KeHttpClient] = KeHttpClient,
) -> int:
    """Run one task strictly through the local HTTP/SSE server boundary."""

    server: ServerLike | None = None
    client: KeHttpClient | None = None
    session_id: str | None = None
    exit_code = 1
    cleanup_failed = False
    try:
        app = app_factory(config, auto_approve=auto_approve)
        server = server_factory(app)
        server.start()
        client = http_client_factory(server.base_url)
        session_id = client.create_session()
        client.send_message(session_id, task)

        for event in client.listen_events(session_id):
            _print_event(event, output)
            if event.event == "tool_confirm":
                if auto_approve:
                    raise HttpClientError(
                        "auto_approve Server 不应产生 tool_confirm"
                    )
                permission_id = event.data.get("permission_id")
                if not isinstance(permission_id, str) or not permission_id:
                    raise HttpClientError("tool_confirm 缺少 permission_id")
                allowed = _confirm(event, input_func)
                client.resolve_permission(session_id, permission_id, allowed)
            elif event.event == "final":
                exit_code = 0
                break
            elif event.event == "error":
                exit_code = 1
                break
        else:
            output("[ERROR] SSE 在任务结束前关闭")
    except KeyboardInterrupt:
        if client is not None and session_id is not None:
            try:
                client.abort(session_id)
            except Exception:
                pass
        output("[ERROR] 用户中止运行")
        exit_code = 130
    except (HttpClientError, EmbeddedServerError) as exc:
        output(f"[ERROR] {exc}")
        exit_code = 1
    except Exception as exc:
        output(f"[ERROR] 运行失败（{type(exc).__name__}）")
        exit_code = 1
    finally:
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
            output("[ERROR] 本地运行资源清理失败")

    return 1 if cleanup_failed and exit_code != 130 else exit_code
