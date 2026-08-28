import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any

import httpx


class HttpClientError(RuntimeError):
    """A safe, user-facing local HTTP client failure."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code


class SseProtocolError(HttpClientError):
    """Raised when the local server returns malformed SSE data."""


@dataclass(frozen=True)
class SseEvent:
    id: int
    event: str
    data: dict[str, Any]


def parse_sse(lines: Iterable[str]) -> Iterator[SseEvent]:
    """Parse the small SSE subset emitted by the ke server."""

    event_id: str | None = None
    event_name: str | None = None
    data_lines: list[str] = []

    def dispatch() -> SseEvent | None:
        nonlocal event_id, event_name, data_lines
        if event_id is None and event_name is None and not data_lines:
            return None
        try:
            parsed_id = int(event_id or "")
        except ValueError:
            raise SseProtocolError("SSE event id 非法") from None
        if not event_name:
            raise SseProtocolError("SSE event 类型缺失")
        try:
            data = json.loads("\n".join(data_lines))
        except json.JSONDecodeError:
            raise SseProtocolError("SSE data 不是合法 JSON") from None
        if not isinstance(data, dict):
            raise SseProtocolError("SSE data 必须是 JSON 对象")
        result = SseEvent(parsed_id, event_name, data)
        event_id = None
        event_name = None
        data_lines = []
        return result

    for raw_line in lines:
        line = raw_line.rstrip("\r\n")
        if not line:
            event = dispatch()
            if event is not None:
                yield event
            continue
        if line.startswith(":"):
            continue
        field, separator, value = line.partition(":")
        if not separator:
            value = ""
        elif value.startswith(" "):
            value = value[1:]
        if field == "id":
            event_id = value
        elif field == "event":
            event_name = value
        elif field == "data":
            data_lines.append(value)

    event = dispatch()
    if event is not None:
        yield event


class KeHttpClient:
    """Thin JSON/SSE client for the six-route local server."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 30.0,
        transport: httpx.BaseTransport | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        self._client = client or httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=timeout,
            transport=transport,
        )
        self._owns_client = client is None

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "KeHttpClient":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    @staticmethod
    def _expect(response: httpx.Response, status: int) -> dict[str, Any]:
        if response.status_code != status:
            message = f"HTTP {response.status_code}"
            try:
                body = response.json()
                if isinstance(body, dict) and isinstance(body.get("error"), str):
                    message += f"：{body['error']}"
            except (ValueError, json.JSONDecodeError):
                pass
            raise HttpClientError(
                message,
                status_code=response.status_code,
            )
        try:
            body = response.json()
        except (ValueError, json.JSONDecodeError):
            raise HttpClientError("HTTP 响应不是合法 JSON") from None
        if not isinstance(body, dict):
            raise HttpClientError("HTTP 响应必须是 JSON 对象")
        return body

    def _post(
        self,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
    ) -> httpx.Response:
        try:
            return self._client.post(path, json=json_body)
        except httpx.HTTPError as exc:
            raise HttpClientError(
                f"HTTP 请求失败（{type(exc).__name__}）"
            ) from None

    def create_session(self) -> str:
        body = self._expect(self._post("/session"), 201)
        session_id = body.get("id")
        if not isinstance(session_id, str) or not session_id:
            raise HttpClientError("创建 session 的响应缺少 id")
        return session_id

    def send_message(self, session_id: str, content: str) -> None:
        self._expect(
            self._post(
                f"/session/{session_id}/message",
                json_body={"content": content},
            ),
            202,
        )

    def resolve_permission(
        self,
        session_id: str,
        permission_id: str,
        allow: bool,
    ) -> None:
        self._expect(
            self._post(
                f"/session/{session_id}/permissions/{permission_id}",
                json_body={"allow": allow},
            ),
            200,
        )

    def abort(self, session_id: str) -> None:
        self._expect(self._post(f"/session/{session_id}/abort"), 202)

    def events(
        self,
        session_id: str,
        *,
        last_event_id: int | None = None,
    ) -> Iterator[SseEvent]:
        headers = {"Accept": "text/event-stream"}
        if last_event_id is not None:
            headers["Last-Event-ID"] = str(last_event_id)
        try:
            with self._client.stream(
                "GET",
                f"/session/{session_id}/events",
                headers=headers,
            ) as response:
                if response.status_code != 200:
                    response.read()
                    self._expect(response, 200)
                yield from parse_sse(response.iter_lines())
        except SseProtocolError:
            raise
        except httpx.HTTPError as exc:
            raise HttpClientError(
                f"SSE 连接失败（{type(exc).__name__}）"
            ) from None

    def listen_events(
        self,
        session_id: str,
        *,
        last_event_id: int | None = None,
        max_reconnects: int = 3,
    ) -> Iterator[SseEvent]:
        reconnects = 0
        while True:
            try:
                for event in self.events(
                    session_id,
                    last_event_id=last_event_id,
                ):
                    if last_event_id is not None and event.id <= last_event_id:
                        continue
                    last_event_id = event.id
                    yield event
                    if event.event in {"final", "error"}:
                        return
            except SseProtocolError:
                raise
            except HttpClientError:
                if reconnects >= max_reconnects:
                    raise
            reconnects += 1
            if reconnects > max_reconnects:
                raise HttpClientError("SSE 在任务结束前意外断开")
