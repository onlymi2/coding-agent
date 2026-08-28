# ke-agent

`ke` 是一个从零实现的本地 coding agent harness，不依赖 LangChain、OpenAI Agents SDK、Claude Agent SDK、AutoGen、CrewAI 等 Agent 框架。模型通过 OpenAI-compatible tool calling 选择动作；文件读写、搜索和命令执行均由本机 Runtime 完成。

## 核心特性

- 自研 Agent Loop，按顺序处理同一轮的多个 tool calls，并将工具成功或失败结果回灌模型。
- 六个本地工具：`read_file`、`write_file`、`edit_file`、`list_dir`、`grep`、`bash`。
- Verification Gate：文件修改后，在首次未经验证的完成声明前提醒模型补充验证证据。
- 多种终止保护：最大轮数、连续工具错误、相同工具与参数的 doom-loop 检测，以及外部 abort。
- Workspace 路径约束：文件路径规范化后必须位于 workspace 内；默认拒绝读取 `.env` 等敏感环境文件，但允许 `.env.example`。
- 工具确认门：`read_file`、`list_dir`、`grep` 自动执行；`write_file`、`edit_file`、`bash` 默认需要人工确认。
- Guarded local command execution：固定 workspace `cwd`、timeout、进程树终止、有界输出以及子进程 API Key 环境变量过滤。
- 三级上下文管理：单条工具输出截断、旧工具结果确定性折叠，以及不带工具的 LLM-assisted semantic summary。
- C/S 架构：Starlette HTTP Server 与 SSE 事件流为 Textual TUI、内置 Web 客户端和 headless `ke run` 提供同一 Runtime。
- OpenAI-compatible LLM 客户端，通过配置切换兼容渠道，不让 Agent Loop 依赖具体模型厂商 SDK 对象。
- FakeLLM、MockTransport 和本地替身支持完全离线的核心测试。

Agent 的运行流程可以映射为 THINK、ACT、OBSERVE、COMPACT、DONE 几个阶段；这些阶段通过事件向客户端暴露，而不是另一套 UI 内 Agent 逻辑。

## 架构

```text
        ke (TUI) / Web / ke run
                    │
                 HTTP/SSE
                    │
      ke serve / Embedded Server Runtime
                    │
                Agent Loop
               ╱          ╲
       Local Tools     OpenAI-compatible LLM
```

裸 `ke`、Web 和 headless `ke run` 都是同一个 Server Runtime 的薄客户端。AgentEvent 事件流是 TUI、Web、headless 客户端和测试共享的运行时事实来源。

## 环境要求与安装

- Python 3.11 或更高版本

创建虚拟环境：

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

## 配置

`ke` 不会自动加载 `.env`，也不依赖 `python-dotenv`。`.env.example` 只是环境变量名称和安全示例模板；`.env` 只能作为本地记录，并必须保持在 `.gitignore` 中。

API Key 只从进程环境变量读取：优先使用 `KE_API_KEY`，也可以使用当前 channel 对应的 `<CHANNEL>_API_KEY`，例如 OpenAI channel 对应 `OPENAI_API_KEY`。不要把密钥写入命令行或 YAML。

常用环境变量：

```powershell
$env:KE_API_KEY="<your-key>"
$env:KE_BASE_URL="<openai-compatible-base-url>"
$env:KE_MODEL="<model>"
$env:KE_CHANNEL="<channel>"
```

`ke.yaml` 只用于保存 channel、base URL、model、workspace、host、port 等非敏感配置，可从 `ke.yaml.example` 开始编写。配置加载器会拒绝 YAML 任意层级中的 API Key、`secret`、`token` 等敏感字段。配置优先级为显式 CLI override、环境变量、`ke.yaml`、内置默认值。

## 使用

```bash
python -m ke --help
python -m ke
python -m ke serve
python -m ke run "你的编程任务"
python -m ke run --yes "你的编程任务"
python -m ke run --yes --workspace examples/demo "你的编程任务"
```

- `python -m ke`：启动 Textual TUI 和内嵌 loopback Server。TUI 会显示可复制的 Web attach URL。
- `python -m ke serve`：启动常驻 HTTP/SSE Server，默认地址为 `http://127.0.0.1:8765/`，该地址同时提供内置 Web 客户端。
- `python -m ke run "任务"`：启动一次性内嵌 Server，并通过 HTTP/SSE 运行 headless 客户端。
- `--yes`：由 Server Runtime 自动批准 `write_file`、`edit_file`、`bash`；默认模式仍会请求人工确认。

Web 可直接创建新 session，也可以通过 `/?session=<id>` attach 到已有 TUI session。客户端只消费 HTTP API 和 SSE，不直接调用 LLM、Agent Loop 或本地工具。

## 安全边界

`bash` 是 guarded local execution，不是操作系统级 sandbox。命令仍以 `shell=True` 在本机执行，因此应只在可信 workspace 和合适的人工确认策略下使用。

Runtime 为 `bash` 提供以下边界：

- `cwd` 固定为当前 workspace。
- `write_file`、`edit_file`、`bash` 默认经过人工确认。
- 命令 timeout，并在超时后终止相应进程树。
- stdout/stderr 统一捕获并进行有界截断。
- 子进程继承 `PATH`、虚拟环境、临时目录等普通环境变量，但移除 `KE_API_KEY` 以及所有名称以 `*_API_KEY` 结尾的环境变量。

文件系统工具只能访问规范化后仍位于 workspace 内的路径。该约束不是完整的操作系统隔离，也不替代容器、虚拟机或最小权限账户。

## Verification Gate

成功执行 `write_file` 或 `edit_file` 会产生验证债务。成功的 `pytest`、`unittest`、`compileall` 或对应 `python/py -m ...` bash 命令可以清除债务；失败的验证不会清除。

如果模型在存在验证债务时直接声明完成，Harness 会向上下文回灌一次运行时提醒，要求运行合适的测试或编译检查。若任务确实无法自动验证，模型可以在下一次回复中说明原因并结束；Harness 不通过 NLP 判断该说明是否“充分”，也不会无限阻止完成。

为避免把 shell fallback 当成成功验证，含 `||`、`;` 或管道的命令不作为验证证据；纯 `&&` 命令链只有在整条命令成功且其中包含受支持的验证命令时才可清除债务。

## 真实 E2E 示例

`examples/demo/` 是真实模型演示的独立 workspace。Git 只保留其中的 `.gitkeep`，Agent 生成的 calculator、测试和缓存均为本地临时文件，不会进入项目源码或测试目录。

在 PowerShell 中设置所需进程环境变量后，从仓库根目录运行：

```powershell
python -m ke run --yes --workspace examples/demo "写一个命令行计算器 calculator.py：支持加减乘除，以及 tests/test_calculator.py，然后运行 pytest 直到通过。"
```

任务结束后人工核对生成文件并独立复验：

```powershell
Test-Path examples/demo/calculator.py
Test-Path examples/demo/tests/test_calculator.py
Push-Location examples/demo
python -m pytest -q
Pop-Location
```

如需重新开始，可只清理 `examples/demo` 中的生成内容并保留 `.gitkeep`；不要在仓库根目录运行 calculator 任务，也不需要创建 `examples/demo/src/` 或第二套 Python package。

## 测试

测试通过 FakeLLM、mock HTTP transport、Starlette TestClient、Textual 测试工具和 fake embedded server 覆盖 Agent Loop、工具、上下文、权限、HTTP/SSE 与客户端行为。默认测试不需要真实模型、真实 API Key 或外部网络。

```bash
python -m pytest -q
```

## 项目结构

```text
.
├── .env.example
├── ke.yaml.example
├── pyproject.toml
├── README.md
├── examples/
│   └── demo/
│       └── .gitkeep
├── src/
│   └── ke/
│       ├── agent/
│       │   ├── context.py
│       │   ├── events.py
│       │   ├── loop.py
│       │   └── prompts.py
│       ├── client/
│       │   ├── http.py
│       │   ├── run.py
│       │   └── tui.py
│       ├── llm/
│       │   ├── client.py
│       │   ├── fake_llm.py
│       │   ├── parse.py
│       │   ├── protocol.py
│       │   └── types.py
│       ├── safety/
│       │   ├── confirm.py
│       │   ├── output.py
│       │   └── sandbox.py
│       ├── server/
│       │   ├── app.py
│       │   ├── runtime.py
│       │   └── static.html
│       ├── tools/
│       │   ├── bash.py
│       │   ├── fs.py
│       │   ├── registry.py
│       │   ├── search.py
│       │   └── types.py
│       ├── __main__.py
│       ├── cli.py
│       └── config.py
└── tests/
```
