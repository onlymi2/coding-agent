# ke-agent

`ke` 是一个计划从零实现的本地 coding agent harness。本仓库当前完成到阶段十一：内置单页 Web 客户端。

当前阶段提供可安装的 Python 包、六个本地工具、事件驱动 Agent Loop、三级上下文压缩、OpenAI 兼容客户端，以及带内置 Web 页面的本地 HTTP/SSE Server。裸 `ke` 会启动 Textual 交互终端；`ke serve` 只启动常驻服务；`ke run` 启动临时 loopback Server，再通过 HTTP/SSE 完成一次 headless 任务。TUI、Web 和 headless run 都是同一 Server Runtime 的薄客户端。

## 环境要求

- Python 3.11 或更高版本
- API Key 只能从进程环境变量读取

## 安装

```bash
python -m venv .venv
```

Windows PowerShell：

```powershell
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

macOS / Linux：

```bash
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

按需复制无密钥的 YAML 示例配置：

```bash
cp ke.yaml.example ke.yaml
```

运行前设置环境变量 `KE_API_KEY`、`KE_BASE_URL`、`KE_MODEL`、`KE_CHANNEL`。`.env` 仅可作为本地配置记录，程序不会自动加载，且必须保持在 `.gitignore` 中。

`.env.example` 只是环境变量名称和安全示例模板，`ke` 本身不会自动读取 `.env`。API Key 只从进程环境变量读取；`ke.yaml` 禁止包含 API Key、`secret` 或 `token` 等敏感字段。本项目不依赖 `python-dotenv`。

## CLI

API Key 只从环境变量读取，例如 `KE_API_KEY` 或当前渠道对应的环境变量；不会从 `ke.yaml` 或命令行参数读取。

```bash
python -m ke --help
python -m ke --version
python -m ke
python -m ke serve
python -m ke run "你的编程任务"
```

安装后也可以直接运行：

```bash
ke --help
ke
ke serve
ke run "创建一个 hello.py 并运行验证"
ke run --yes "创建一个 hello.py 并运行验证"
ke run --yes --workspace examples/demo "写一个计算器"
```

裸 `ke` 启动 Textual 交互终端，并通过内嵌 Server 的 HTTP/SSE 接口工作。TUI 日志会显示一个可复制的 `WEB http://127.0.0.1:<port>/?session=<id>` 地址；在浏览器打开它会 attach 到同一个 session，两端观察同一串 Agent 事件。

`ke serve` 只启动 HTTP/SSE 服务，默认监听 `127.0.0.1:8765`。浏览器访问 `http://127.0.0.1:8765/` 时，Web 客户端会创建自己的 session；访问 `http://127.0.0.1:8765/?session=<id>` 时则 attach 已有 session。网页只使用现有 HTTP API 和 SSE，不直接访问模型、工具或文件系统。

普通 `ke run` 是一次性 headless 客户端，遇到写文件、编辑文件或执行命令时询问 `y/N`，默认拒绝；`ke run --yes` 让内嵌 Server 使用 `auto_approve=True`。一次性 run 使用动态 loopback 端口，结束或 Ctrl+C 后关闭内嵌 Server。

## 真实模型演示准备

真实 calculator 演示只使用 `examples/demo/` 作为 workspace，不要在仓库根目录运行演示任务。程序不会自动加载 `.env`；在 PowerShell 中由当前进程显式设置配置：

```powershell
$env:KE_API_KEY="<your-key>"
$env:KE_BASE_URL="<openai-compatible-base-url>"
$env:KE_MODEL="<model>"
$env:KE_CHANNEL="<channel>"
```

`.env` 仍然只能作为本地记录并保持在 `.gitignore` 中，`.env.example` 只是模板。API Key 只从进程环境变量读取，`ke.yaml` 禁止保存 API Key、`secret` 或 `token`，项目不会引入或调用 `python-dotenv`。

录像或试跑前，从仓库根目录执行以下固定范围的 PowerShell reset。它只清理 `examples/demo` 内可能由上一次演示生成的文件，并保留 `.gitkeep`：

```powershell
Remove-Item -LiteralPath examples/demo/calculator.py -Force -ErrorAction SilentlyContinue
Remove-Item -LiteralPath examples/demo/tests -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item -LiteralPath examples/demo/.pytest_cache -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item -LiteralPath examples/demo/__pycache__ -Recurse -Force -ErrorAction SilentlyContinue
```

演示命令：

```powershell
python -m ke run --yes --workspace examples/demo "写一个命令行计算器 calculator.py：支持加减乘除，以及 tests/test_calculator.py，然后运行 pytest 直到通过。"
```

这里不创建 `examples/demo/src/` 或第二套 Python package。任务中的 `calculator.py`、`tests/test_calculator.py` 和 `pytest -q` 都以 `examples/demo` 为唯一根目录；演示前该目录只保留 `.gitkeep`，不预置答案。

真实任务结束后，人工核对：

```powershell
Test-Path examples/demo/calculator.py
Test-Path examples/demo/tests/test_calculator.py
Push-Location examples/demo
pytest -q
Pop-Location
```

第一次真实运行只作为稳定性试跑。建议每次先 reset，连续试跑 2～3 次；确认模型能稳定完成写文件、执行 pytest、根据失败修正并最终 DONE 后，再 reset 一次进行正式录像。

## 当前结构

```text
.
├── pyproject.toml
├── .env.example
├── ke.yaml.example
├── examples/
│   └── demo/
│       └── .gitkeep
└── src/
    └── ke/
        ├── __init__.py
        ├── __main__.py
        ├── cli.py
        ├── config.py
        ├── safety/
        │   ├── sandbox.py
        │   ├── output.py
        │   └── confirm.py
        ├── tools/
        │   ├── types.py
        │   ├── fs.py
        │   ├── search.py
        │   ├── bash.py
        │   └── registry.py
        ├── agent/
        │   ├── context.py
        │   ├── events.py
        │   ├── loop.py
        │   └── prompts.py
        ├── server/
        │   ├── __init__.py
        │   ├── app.py
        │   ├── static.html
        │   └── runtime.py
        ├── client/
        │   ├── __init__.py
        │   ├── http.py
        │   ├── run.py
        │   └── tui.py
        └── llm/
            ├── types.py
            ├── protocol.py
            ├── parse.py
            ├── fake_llm.py
            └── client.py
```

后续能力会严格按照教程分阶段加入。配置优先级为显式覆盖、环境变量、`ke.yaml`、内置默认值；API Key 只允许来自环境变量。阶段十一测试使用 Starlette TestClient、Textual `run_test()`、Pilot、MockTransport、FakeLLM、fake embedded server 和 mock uvicorn，不读取真实 `.env`、不调用真实模型，也不访问外部网络。
