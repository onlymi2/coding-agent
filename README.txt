KE：一个可验证、可终止、可观察的本地Coding Agent

1、Git仓库
　　https://github.com/onlymi2/coding-agent

2、项目简介
　　KE是一个从零实现的本地Coding Agent。名称取“壳”之意：它不是模型本身，而是模型外部的Agent Harness，负责将模型的决策转化为真实的文件读写、命令执行与环境反馈。
　　KE自行实现Agent Loop、上下文管理、本地工具执行、验证与终止逻辑，不依赖LangChain、OpenAI Agents SDK等现成Agent框架。工具执行结果会作为Observation返回模型，使其基于真实环境反馈持续修改和验证代码，直至任务完成或触发终止条件。

3、运行方法
（1）安装
　　需要Python 3.11及以上。在项目根目录创建虚拟环境并安装：
　　python -m venv .venv
　　.venv\Scripts\Activate.ps1
　　python -m pip install -e ".[dev]"

（2）配置模型
　　KE通过OpenAI-compatible接口接入模型。以PowerShell为例：
　　$env:KE_API_KEY="<your-api-key>"
　　$env:KE_BASE_URL="<openai-compatible-base-url>"
　　$env:KE_MODEL="<model-name>"
　　$env:KE_CHANNEL="<channel-name>"
　　API Key仅通过环境变量提供，不写入仓库或YAML配置文件。

（3）运行方式
　　python -m ke
　　启动Textual TUI及内嵌本地Server，适合交互使用；Web客户端可连接同一运行会话。
　　python -m ke serve
　　单独启动HTTP/SSE Server及内置Web客户端，默认访问http://127.0.0.1:8765/。
　　python -m ke run --yes "你的编程任务"
　　启动一次性headless任务，通过内嵌Server执行；--yes用于自动批准文件修改和命令执行。

4、核心设计与特色
（1）Verification Gate：一次性完成前验证提醒
　　代码经write_file或edit_file修改后进入待验证状态。若模型在没有新验证证据时请求结束，Runtime会拦截首次完成并提醒验证；成功执行pytest、unittest或compileall可清除待验证状态。确实无法自动验证时，可说明原因后结束。这样避免模型修改代码后立即宣称完成，同时保留无法自动验证时的退出路径。

（2）Bounded Loop：有界执行
　　Agent运行过程可映射为THINK / ACT / OBSERVE / COMPACT / DONE阶段，工具成功或失败都会作为Observation回到上下文，使模型基于真实执行结果继续决策。Runtime同时设置最大轮数、连续工具错误、重复相同动作的doom-loop检测和用户abort等终止条件，使错误可以被Agent消化，也保证循环能够有界终止。

（3）Workspace Safety：执行边界
　　所有文件工具路径规范化后必须位于workspace内，并默认拒绝读取.env等敏感文件；write_file、edit_file和bash默认需要人工确认。bash固定在workspace执行，并提供timeout、进程终止、输出截断和API Key环境变量过滤。该机制属于受约束的本地执行，并非操作系统级沙箱。

（4）Unified Event Runtime：统一事件流
　　Agent Core不直接依赖具体界面，只产生结构化运行事件。同一个HTTP/SSE Runtime同时服务Textual TUI、内置Web和headless客户端，三种入口共享同一套Agent Loop与执行状态，使界面可以独立替换，也便于测试和观察完整运行过程。

5、其他说明
（1）Context Management：三级上下文管理
　　KE对上下文进行分级压缩：先截断超长工具输出，再折叠较早的工具结果，必要时由不带工具的LLM生成历史摘要，在控制上下文长度的同时尽量保留任务目标和有效信息。

（2）模型解耦与离线测试
　　Agent Loop通过LlmClient协议接入OpenAI-compatible模型，可通过配置切换不同模型与网关。核心逻辑使用FakeLLM等本地替身进行离线测试，不依赖真实API Key和外部网络。

（3）实验验证
　　安装dev依赖后运行python -m pytest -q，实际收集236个测试用例，236/236通过、0 failed、0 skipped；其中Verification Gate相关用例23/23通过。全部测试无需真实LLM、API Key或外部网络。
