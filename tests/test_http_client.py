import json

import httpx
import pytest

from ke.client.http import (
    HttpClientError,
    KeHttpClient,
    SseProtocolError,
    parse_sse,
)


def test_json_endpoints_use_expected_routes_and_payloads() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/session":
            return httpx.Response(201, json={"id": "session-1"})
        if request.url.path.endswith("/message"):
            return httpx.Response(202, json={"status": "accepted"})
        if "/permissions/" in request.url.path:
            return httpx.Response(
                200,
                json={"permission_id": "permission-1", "allowed": True},
            )
        if request.url.path.endswith("/abort"):
            return httpx.Response(202, json={"status": "aborting"})
        return httpx.Response(404, json={"error": "missing"})

    with KeHttpClient(
        "http://test",
        transport=httpx.MockTransport(handler),
    ) as client:
        session_id = client.create_session()
        client.send_message(session_id, "task")
        client.resolve_permission(session_id, "permission-1", True)
        client.abort(session_id)

    assert session_id == "session-1"
    assert [request.url.path for request in requests] == [
        "/session",
        "/session/session-1/message",
        "/session/session-1/permissions/permission-1",
        "/session/session-1/abort",
    ]
    assert json.loads(requests[1].content) == {"content": "task"}
    assert json.loads(requests[2].content) == {"allow": True}


def test_send_message_requires_202() -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(409, json={"error": "busy"})
    )
    with KeHttpClient("http://test", transport=transport) as client:
        with pytest.raises(HttpClientError, match="409") as exc_info:
            client.send_message("session", "task")

    assert exc_info.value.status_code == 409


def test_parse_sse_ignores_keep_alive_and_parses_json() -> None:
    events = list(
        parse_sse(
            [
                ": keep-alive",
                "",
                "id: 7",
                "event: final",
                'data: {"type":"final","message":"done"}',
                "",
            ]
        )
    )

    assert len(events) == 1
    assert events[0].id == 7
    assert events[0].event == "final"
    assert events[0].data == {"type": "final", "message": "done"}


def test_events_passes_last_event_id_header() -> None:
    captured: dict[str, str | None] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["last_event_id"] = request.headers.get("last-event-id")
        return httpx.Response(
            200,
            text='id: 4\nevent: final\ndata: {"type":"final"}\n\n',
            headers={"content-type": "text/event-stream"},
        )

    with KeHttpClient(
        "http://test",
        transport=httpx.MockTransport(handler),
    ) as client:
        events = list(client.events("session", last_event_id=3))

    assert captured["last_event_id"] == "3"
    assert [event.id for event in events] == [4]


def test_listen_events_reconnects_with_latest_event_id() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            text = 'id: 1\nevent: turn_start\ndata: {"type":"turn_start"}\n\n'
        else:
            text = 'id: 2\nevent: final\ndata: {"type":"final"}\n\n'
        return httpx.Response(
            200,
            text=text,
            headers={"content-type": "text/event-stream"},
        )

    with KeHttpClient(
        "http://test",
        transport=httpx.MockTransport(handler),
    ) as client:
        events = list(client.listen_events("session"))

    assert [event.id for event in events] == [1, 2]
    assert requests[0].headers.get("last-event-id") is None
    assert requests[1].headers["last-event-id"] == "1"


def test_listen_events_accepts_an_initial_cursor() -> None:
    captured: dict[str, str | None] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["last_event_id"] = request.headers.get("last-event-id")
        return httpx.Response(
            200,
            text='id: 6\nevent: final\ndata: {"type":"final"}\n\n',
            headers={"content-type": "text/event-stream"},
        )

    with KeHttpClient(
        "http://test",
        transport=httpx.MockTransport(handler),
    ) as client:
        events = list(
            client.listen_events("session", last_event_id=5)
        )

    assert captured["last_event_id"] == "5"
    assert [event.id for event in events] == [6]


@pytest.mark.parametrize(
    ("lines", "message"),
    [
        (["id: nope", "event: final", "data: {}", ""], "id 非法"),
        (["id: 1", "data: {}", ""], "类型缺失"),
        (["id: 1", "event: final", "data: nope", ""], "合法 JSON"),
        (["id: 1", "event: final", "data: []", ""], "JSON 对象"),
    ],
)
def test_malformed_sse_has_clear_error(
    lines: list[str],
    message: str,
) -> None:
    with pytest.raises(SseProtocolError, match=message):
        list(parse_sse(lines))
