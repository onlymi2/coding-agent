import re
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
        "event instanceof MessageEvent",
    ):
        assert marker in html


def test_static_html_maps_existing_events_to_agent_phase_graph() -> None:
    html = load_static_html()

    for phase in ("THINK", "ACT", "OBSERVE", "COMPACT", "DONE"):
        assert f'data-phase="{phase}"' in html
    assert html.count('class="phase-node" data-phase=') == 5
    assert 'data-phase="CONFIRM"' not in html

    mappings = (
        ("turn_start", "THINK"),
        ("tool_request", "ACT"),
        ("tool_confirm", "ACT"),
        ("tool_result", "OBSERVE"),
        ("context_compact", "COMPACT"),
        ("context_summary", "COMPACT"),
        ("final", "DONE"),
    )
    for event_name, phase in mappings:
        pattern = (
            rf'name === "{event_name}".*?'
            rf'setPhase\("{phase}"\)'
        )
        assert re.search(pattern, html, re.DOTALL)

    error_branch = re.search(
        r'} else if \(name === "error"\) \{(?P<body>.*?)\n      \}\n    \}',
        html,
        re.DOTALL,
    )
    assert error_branch is not None
    assert "setPhaseError();" in error_branch.group("body")
    assert 'setPhase("DONE")' not in error_branch.group("body")


def test_static_html_keeps_desktop_layout_inside_viewport() -> None:
    html = load_static_html()

    for rule in (
        "min-height: 100dvh;",
        "overflow: hidden;",
        "height: calc(100dvh - 32px);",
        "grid-template-rows: auto minmax(0, 1fr) auto;",
        "main > section.panel { display: flex; flex-direction: column; }",
        "#events {",
        "height: 100%;",
        "flex: 1;",
        "overflow-y: auto;",
    ):
        assert rule in html

    mobile = html.split("@media (max-width: 780px)", 1)[1]
    assert "body { min-height: 100dvh; overflow: auto; }" in mobile
    assert ".shell { height: auto;" in mobile


def test_static_html_hides_permission_and_abort_outside_active_work() -> None:
    html = load_static_html()

    assert '#permission { display: none;' in html
    assert '#abort { display: none; }' in html
    assert 'ui.permission.style.display = "block";' in html
    assert 'function hidePermission()' in html
    assert 'ui.permission.style.display = "none";' in html
    assert 'ui.abort.style.display = value ? "block" : "none";' in html

    assert re.search(
        r'name === "tool_result".*?hidePermission\(\).*?'
        r'name === "context_compact"',
        html,
        re.DOTALL,
    )
    assert re.search(
        r'name === "final".*?clearTool\(\).*?setRunning\(false\).*?'
        r'name === "error"',
        html,
        re.DOTALL,
    )
    assert re.search(
        r'name === "error".*?clearTool\(\).*?setRunning\(false\).*?'
        r'\n      \}\n    \}',
        html,
        re.DOTALL,
    )
    assert re.search(
        r'async function createSession\(clearLog\).*?clearTool\(\)',
        html,
        re.DOTALL,
    )

    assert "resolvePermission(false)" in html
    assert "resolvePermission(true)" in html
    assert "error.status === 409" in html
    assert "该权限已由其他客户端处理" in html


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
    assert "function isagentmessageevent(event)" in lowered
    assert "source === current && isagentmessageevent(event)" in lowered
    assert "current.onerror = function (event)" in lowered
    assert "if (isagentmessageevent(event))" in lowered
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
