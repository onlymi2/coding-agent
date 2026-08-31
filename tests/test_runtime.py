import threading
import time
from types import SimpleNamespace
from pathlib import Path

import httpx
import pytest

import ke.server.runtime as runtime_module
from ke.config import KeConfig
from ke.llm.fake_llm import FakeLLM
from ke.llm.types import LLMResponse, Message
from ke.server.app import create_app
from ke.server.runtime import EmbeddedServerError


class FakeSocket:
    def __init__(self) -> None:
        self.bound_to: tuple[str, int] | None = None
        self.backlog: int | None = None
        self.closed = threading.Event()
        self.close_calls = 0

    def setsockopt(self, *args: object) -> None:
        pass

    def bind(self, address: tuple[str, int]) -> None:
        self.bound_to = address

    def listen(self, backlog: int) -> None:
        self.backlog = backlog

    def getsockname(self) -> tuple[str, int]:
        return "127.0.0.1", 43123

    def close(self) -> None:
        self.close_calls += 1
        self.closed.set()


class CooperativeUvicornServer:
    def __init__(self, config: object) -> None:
        self.started = False
        self.force_exit = False
        self._should_exit = False
        self._exit = threading.Event()
        self.sockets: list[FakeSocket] = []

    @property
    def should_exit(self) -> bool:
        return self._should_exit

    @should_exit.setter
    def should_exit(self, value: bool) -> None:
        self._should_exit = value
        if value:
            self._exit.set()

    def run(self, *, sockets: list[FakeSocket]) -> None:
        self.sockets = sockets
        self.started = True
        self._exit.wait(timeout=2)


class FailingUvicornServer:
    def __init__(self, config: object) -> None:
        self.started = False
        self.should_exit = False
        self.force_exit = False

    def run(self, *, sockets: list[FakeSocket]) -> None:
        raise RuntimeError("fake startup failure")


class SocketBoundUvicornServer:
    def __init__(self, config: object) -> None:
        self.started = True
        self.should_exit = False
        self.force_exit = False

    def run(self, *, sockets: list[FakeSocket]) -> None:
        sockets[0].closed.wait(timeout=2)


class NeverStartsUvicornServer(SocketBoundUvicornServer):
    def __init__(self, config: object) -> None:
        super().__init__(config)
        self.started = False


def install_embedded_fakes(
    monkeypatch: pytest.MonkeyPatch,
    server_type: type,
) -> tuple[FakeSocket, list[object]]:
    listener = FakeSocket()
    servers: list[object] = []
    fake_socket_module = SimpleNamespace(
        AF_INET=2,
        SOCK_STREAM=1,
        SOL_SOCKET=1,
        SO_REUSEADDR=2,
        socket=lambda *args: listener,
    )

    def server_factory(config: object) -> object:
        server = server_type(config)
        servers.append(server)
        return server

    monkeypatch.setattr(runtime_module, "socket", fake_socket_module)
    monkeypatch.setattr(runtime_module.uvicorn, "Config", lambda *args, **kwargs: object())
    monkeypatch.setattr(runtime_module.uvicorn, "Server", server_factory)
    return listener, servers


def make_config(tmp_path: Path) -> KeConfig:
    return KeConfig(
        channel="test-channel",
        base_url="https://provider.example/v1",
        model="test-model",
        api_key="test-key",
        workspace=tmp_path,
        host="127.0.0.2",
        port=9012,
    )


def test_production_llm_factory_is_lazy_and_uses_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[dict[str, object]] = []

    class FakeClient:
        def __init__(self, **kwargs: object) -> None:
            captured.append(kwargs)

    monkeypatch.setattr(runtime_module, "OpenAICompatClient", FakeClient)
    factory = runtime_module.build_llm_factory(make_config(tmp_path))

    assert captured == []
    factory()
    assert captured == [
        {
            "api_key": "test-key",
            "base_url": "https://provider.example/v1",
            "model": "test-model",
        }
    ]


@pytest.mark.parametrize("auto_approve", [False, True])
def test_serve_uses_uvicorn_config_without_network_or_llm_call(
    auto_approve: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_uvicorn_run(app: object, **kwargs: object) -> None:
        captured.update(app=app, **kwargs)

    monkeypatch.setattr(runtime_module.uvicorn, "run", fake_uvicorn_run)
    config = make_config(tmp_path)

    runtime_module.serve(config, auto_approve=auto_approve)

    app = captured["app"]
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
    assert app.state.runtime.auto_approve is auto_approve
    assert captured["host"] == "127.0.0.2"
    assert captured["port"] == 9012


def test_embedded_server_starts_on_prebound_loopback_socket_and_stops_twice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    listener, servers = install_embedded_fakes(
        monkeypatch,
        CooperativeUvicornServer,
    )
    embedded = runtime_module.EmbeddedServer(object())

    embedded.start()

    assert embedded.base_url == "http://127.0.0.1:43123"
    assert listener.bound_to == ("127.0.0.1", 0)
    assert listener.backlog == 2048
    assert servers[0].started

    embedded.stop()
    embedded.stop()

    assert listener.close_calls == 1
    assert embedded._socket is None
    assert embedded._thread is not None
    assert not embedded._thread.is_alive()


def test_embedded_server_start_failure_closes_listener(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    listener, _ = install_embedded_fakes(
        monkeypatch,
        FailingUvicornServer,
    )
    embedded = runtime_module.EmbeddedServer(object())

    with pytest.raises(EmbeddedServerError, match="启动失败"):
        embedded.start()

    assert listener.closed.is_set()
    assert embedded._socket is None
    assert embedded._thread is not None
    assert not embedded._thread.is_alive()


def test_embedded_server_start_timeout_cleans_thread_and_socket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    listener, _ = install_embedded_fakes(
        monkeypatch,
        NeverStartsUvicornServer,
    )
    embedded = runtime_module.EmbeddedServer(
        object(),
        startup_timeout=0.02,
        shutdown_timeout=0.01,
    )

    with pytest.raises(EmbeddedServerError, match="启动超时"):
        embedded.start()

    assert listener.closed.is_set()
    assert embedded._socket is None
    assert embedded._thread is not None
    assert not embedded._thread.is_alive()


def test_embedded_stop_closes_socket_then_gives_thread_final_join(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    listener, servers = install_embedded_fakes(
        monkeypatch,
        SocketBoundUvicornServer,
    )
    embedded = runtime_module.EmbeddedServer(
        object(),
        shutdown_timeout=0.01,
    )
    embedded.start()

    embedded.stop()

    assert servers[0].force_exit is True
    assert listener.closed.is_set()
    assert embedded._thread is not None
    assert not embedded._thread.is_alive()


@pytest.mark.parametrize("after_first_task", [False, True])
def test_embedded_stop_closes_idle_session_sse_subscriber(
    after_first_task: bool,
    tmp_path: Path,
) -> None:
    app = create_app(
        make_config(tmp_path),
        llm_factory=lambda: FakeLLM(
            [
                LLMResponse(
                    Message(role="assistant", content="done"),
                    finish_reason="stop",
                )
            ]
        ),
    )
    embedded = runtime_module.EmbeddedServer(
        app,
        startup_timeout=3,
        shutdown_timeout=3,
    )
    client: httpx.Client | None = None
    subscriber: threading.Thread | None = None
    connected = threading.Event()
    stopped = threading.Event()
    events: list[str] = []
    errors: list[str] = []

    try:
        embedded.start()
        client = httpx.Client(
            base_url=embedded.base_url,
            timeout=3,
            trust_env=False,
        )
        response = client.post("/session")
        assert response.status_code == 201
        session_id = response.json()["id"]

        def observe() -> None:
            try:
                with httpx.Client(
                    base_url=embedded.base_url,
                    timeout=None,
                    trust_env=False,
                ) as observer:
                    with observer.stream(
                        "GET", f"/session/{session_id}/events"
                    ) as stream:
                        assert stream.status_code == 200
                        connected.set()
                        for line in stream.iter_lines():
                            if line.startswith("event: "):
                                events.append(line.removeprefix("event: "))
            except Exception as exc:  # pragma: no cover - assertion below
                errors.append(type(exc).__name__)
            finally:
                stopped.set()

        subscriber = threading.Thread(target=observe, daemon=True)
        subscriber.start()
        assert connected.wait(timeout=2)
        assert subscriber.is_alive()

        if after_first_task:
            response = client.post(
                f"/session/{session_id}/message",
                json={"content": "task A"},
            )
            assert response.status_code == 202
            deadline = time.monotonic() + 3
            while "final" not in events and time.monotonic() < deadline:
                time.sleep(0.01)
            assert "final" in events
            assert subscriber.is_alive()

        embedded.stop()
        assert stopped.wait(timeout=2)
        subscriber.join(timeout=1)
        assert not subscriber.is_alive()
        assert errors == []
    finally:
        if client is not None:
            client.close()
        try:
            embedded.stop()
        except EmbeddedServerError:
            pass
