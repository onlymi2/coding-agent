import socket
import threading
import time
from collections.abc import Callable

import uvicorn
from starlette.applications import Starlette

from ke.config import KeConfig
from ke.llm.client import OpenAICompatClient
from ke.llm.protocol import LlmClient
from ke.server.app import create_app


class EmbeddedServerError(RuntimeError):
    """Raised when the one-shot local server cannot start or stop cleanly."""


def build_llm_factory(config: KeConfig) -> Callable[[], LlmClient]:
    """Build a lazy, provider-independent session LLM factory."""

    def factory() -> LlmClient:
        return OpenAICompatClient(
            api_key=config.api_key,
            base_url=config.base_url,
            model=config.model,
        )

    return factory


def build_app(config: KeConfig, *, auto_approve: bool = False) -> Starlette:
    return create_app(
        config,
        llm_factory=build_llm_factory(config),
        auto_approve=auto_approve,
    )


def serve(config: KeConfig, *, auto_approve: bool = False) -> None:
    """Run the configured six-route server until uvicorn exits."""

    uvicorn.run(
        build_app(config, auto_approve=auto_approve),
        host=config.host,
        port=config.port,
    )


class EmbeddedServer:
    """Background uvicorn server bound atomically to a loopback port."""

    def __init__(
        self,
        app: Starlette,
        *,
        startup_timeout: float = 5.0,
        shutdown_timeout: float = 5.0,
    ) -> None:
        self.app = app
        self.startup_timeout = startup_timeout
        self.shutdown_timeout = shutdown_timeout
        self._socket: socket.socket | None = None
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None
        self._stopped = threading.Event()
        self._failure: str | None = None
        self._port: int | None = None

    @property
    def base_url(self) -> str:
        if self._port is None:
            raise EmbeddedServerError("内嵌服务尚未启动")
        return f"http://127.0.0.1:{self._port}"

    def start(self) -> None:
        if self._thread is not None:
            raise EmbeddedServerError("内嵌服务不能重复启动")

        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(("127.0.0.1", 0))
            listener.listen(2048)
        except OSError as exc:
            listener.close()
            raise EmbeddedServerError(
                f"内嵌服务绑定失败（{type(exc).__name__}）"
            ) from None

        self._socket = listener
        self._port = int(listener.getsockname()[1])
        config = uvicorn.Config(
            self.app,
            log_level="warning",
            access_log=False,
            lifespan="off",
        )
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(
            target=self._run,
            name="ke-embedded-server",
            daemon=True,
        )
        self._thread.start()

        deadline = time.monotonic() + self.startup_timeout
        while not self._server.started:
            if self._stopped.wait(timeout=0.01):
                self.stop()
                detail = self._failure or "未知错误"
                raise EmbeddedServerError(f"内嵌服务启动失败（{detail}）")
            if time.monotonic() >= deadline:
                self.stop()
                raise EmbeddedServerError("内嵌服务启动超时")

    def _run(self) -> None:
        assert self._server is not None
        assert self._socket is not None
        try:
            self._server.run(sockets=[self._socket])
        except Exception as exc:
            self._failure = type(exc).__name__
        finally:
            self._stopped.set()

    def stop(self) -> None:
        server = self._server
        thread = self._thread
        runtime = getattr(getattr(self.app, "state", None), "runtime", None)
        shutdown = getattr(runtime, "shutdown", None)
        if callable(shutdown):
            shutdown()
        if server is not None:
            server.should_exit = True
        if thread is not None and thread.is_alive():
            thread.join(timeout=self.shutdown_timeout)
            if thread.is_alive() and server is not None:
                server.force_exit = True
        if self._socket is not None:
            try:
                self._socket.close()
            except OSError:
                pass
            finally:
                self._socket = None
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)
        if thread is not None and thread.is_alive():
            raise EmbeddedServerError("内嵌服务线程未能停止")
