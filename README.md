# ke-agent

`ke` 是一个计划从零实现的本地 coding agent harness。本仓库当前完成到阶段八：HTTP Server、SSE 事件广播与工具权限确认门。

当前阶段提供可安装的 Python 包、消息与响应边界、六个本地工具、工具注册表、事件驱动 Agent Loop、三级上下文压缩，以及厂商无关的 OpenAI 兼容 Chat Completions 客户端。薄 HTTP 层提供六个固定接口；Agent 在后台线程运行，事件通过可回放的 SSE 广播。`write_file`、`edit_file` 和 `bash` 默认等待人工确认，其他只读工具自动执行。真实客户端尚未接入默认 CLI 流程；测试继续使用 FakeLLM，完全离线。尚未实现阶段九 CLI 接线、TUI、静态网页或持久化 session。

## 环境要求

- Python 3.11 或更高版本
- API Key 只能放在环境变量或未入库的 `.env` 中

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

按需复制示例配置：

```bash
cp .env.example .env
cp ke.yaml.example ke.yaml
```

不要把真实 API Key 写入 `.env.example` 或 `ke.yaml`。

## CLI

```bash
python -m ke --help
python -m ke --version
```

安装后也可以直接运行：

```bash
ke --help
```

## 当前结构

```text
.
├── pyproject.toml
├── .env.example
├── ke.yaml.example
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
        │   └── app.py
        └── llm/
            ├── types.py
            ├── protocol.py
            ├── parse.py
            ├── fake_llm.py
            └── client.py
```

后续能力会严格按照教程分阶段加入。上下文依次执行单条工具输出截断、旧工具结果折叠和旧中段语义摘要；system、原始任务及最近工具轮次始终保留。配置优先级为显式覆盖、环境变量、`ke.yaml`、内置默认值；API Key 只允许来自环境变量。本阶段测试不读取真实 `.env`、不调用真实模型，也不访问网络；Agent Loop 和 Context 都只依赖 `LlmClient` 协议。
