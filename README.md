# ke-agent

`ke` 是一个计划从零实现的本地 coding agent harness。本仓库当前只完成教程的阶段一：工程骨架。

当前阶段提供可安装的 Python 包、`ke` 命令入口、示例配置和基础项目约定。尚未实现 Agent Loop、Tools、Context、TUI 或 Server。

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
        └── cli.py
```

后续能力会严格按照教程分阶段加入，本阶段不包含任何真实模型调用或本地工具执行。
