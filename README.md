# ke-agent

`ke` 是一个计划从零实现的本地 coding agent harness。本仓库当前完成到教程阶段六：Context 上下文管理与确定性压缩。

当前阶段提供可安装的 Python 包、`ke` 命令入口、消息与响应边界、六个本地工具、工具注册表、事件驱动 Agent Loop，以及独立的上下文估算与两层确定性压缩。循环只依赖 `LlmClient` 协议，不依赖具体模型 SDK。尚未实现 LLM 摘要压缩、真实 LLM 网络请求、配置系统、TUI 或 Server。

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
        ├── safety/
        │   ├── sandbox.py
        │   └── output.py
        ├── tools/
        │   ├── types.py
        │   ├── fs.py
        │   ├── search.py
        │   ├── bash.py
        │   └── registry.py
        ├── agent/
        │   ├── context.py
        │   ├── events.py
        │   └── loop.py
        └── llm/
            ├── types.py
            ├── protocol.py
            ├── parse.py
            └── fake_llm.py
```

后续能力会严格按照教程分阶段加入。本阶段使用 FakeLLM 验证压缩后的完整循环，不读取真实 `.env`、不调用真实模型，也不访问网络。Context 使用启发式 token 估算，保留 system、用户任务、assistant 工具调用结构和最近工具结果；Agent Loop 不直接打印，全部过程通过事件暴露。
