from collections.abc import Sequence
from pathlib import Path

from starlette.testclient import TestClient

from ke.config import KeConfig
from ke.llm.fake_llm import FakeLLM
from ke.llm.protocol import ToolSchema
from ke.llm.types import LLMResponse, Message, ToolCall
from ke.server.app import create_app, load_static_html


def make_config(tmp_path: Path) -> KeConfig:
    return KeConfig(
        channel="test",
        base_url="https://unused.example/v1",
        model="fake-model",
        api_key="test-key-must-not-appear",
        workspace=tmp_path,
    )


def final_response(content: str = "done") -> LLMResponse:
    return LLMResponse(
        message=Message(role="assistant", content=content),
        finish_reason="stop",
    )


def tool_response() -> LLMResponse:
    return LLMResponse(
        message=Message(
            role="assistant",
            tool_calls=[ToolCall("list", "list_dir", {"path": "."})],
        ),
        finish_reason="tool_calls",
    )


def test_root_serves_packaged_html_without_initializing_llm(
    tmp_path: Path,
) -> None:
    factory_calls: list[int] = []

    def factory() -> FakeLLM:
        factory_calls.append(1)
        return FakeLLM([final_response()])

    app = create_app(make_config(tmp_path), llm_factory=factory)
    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert response.text == load_static_html()
    assert factory_calls == []
    assert "test-key-must-not-appear" not in response.text


def test_server_keeps_six_business_apis_plus_one_web_route(
    tmp_path: Path,
) -> None:
    app = create_app(
        make_config(tmp_path),
        llm_factory=lambda: FakeLLM([final_response()]),
    )

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


def test_static_html_contains_the_thin_http_sse_client() -> None:
    html = load_static_html()

    for marker in (
        'id="task-input"',
        ">Send<",
        ">Abort<",
        ">New Session<",
        "new EventSource(",
        "fetch(",
        'new URLSearchParams(window.location.search).get("session")',
        '"turn_start"',
        '"tool_request"',
        '"tool_confirm"',
        '"tool_result"',
        '"context_compact"',
        '"context_summary"',
        '"final"',
        '"error"',
        "lastEventId",
        "terminalSeen",
    ):
        assert marker in html


def test_static_html_uses_safe_text_rendering_and_no_external_frontend() -> None:
    html = load_static_html()
    lowered = html.casefold()

    assert "innerhtml" not in lowered
    assert ".textcontent" in lowered
    assert "document.createelement" in lowered
    assert "api_key" not in lowered
    assert "authorization" not in lowered
    assert "<script src=" not in lowered
    assert "<link rel=" not in lowered
    assert "https://" not in lowered
    assert "terminaltimer" not in lowered
    assert "scheduleterminalclose" not in lowered
    assert "settimeout" not in lowered
    assert "if (terminalseen)" in lowered
    for framework in (
        "react.production",
        "vue.global",
        "vite/client",
        "jquery",
        "bootstrap.min",
    ):
        assert framework not in lowered


class RecordingLLM:
    def __init__(self) -> None:
        self.fake = FakeLLM([tool_response(), final_response()])

    def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSchema],
    ) -> LLMResponse:
        return self.fake.complete(messages, tools)


def test_same_session_history_is_broadcast_to_two_web_observers(
    tmp_path: Path,
) -> None:
    app = create_app(make_config(tmp_path), llm_factory=RecordingLLM)
    with TestClient(app) as client:
        session_id = client.post("/session").json()["id"]
        accepted = client.post(
            f"/session/{session_id}/message",
            json={"content": "list"},
        )
        assert accepted.status_code == 202
        session = app.state.runtime.get(session_id)
        assert session is not None
        assert session.wait_until_idle()

        first = client.get(f"/session/{session_id}/events")
        second = client.get(f"/session/{session_id}/events")

    assert first.status_code == 200
    assert first.text == second.text
    for event_name in (
        "turn_start",
        "tool_request",
        "tool_result",
        "final",
    ):
        assert f"event: {event_name}" in first.text
