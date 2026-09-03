# AGENTS.md

本代码库以简洁与可读性优先。
请编写 Python 初学者也能看懂、维护和扩展的代码。

## 原则 (Principles)

1.  **清晰至上 (Clarity first)**
    - 编写简单、清晰、易于理解的代码。
    - 避免过早优化和不必要的抽象。
    - 遵循《Python 之禅》(The Zen of Python) 。
    - 目标版本为 **Python 3.12+**；避免使用已弃用的写法。

2.  **单一职责 (Single responsibility)**
    - 每个函数/类/模块都应当只把 **一件事** 做好。

3.  **关注性能 (Performance-aware)**
    - 在关键路径上对性能保持敏感。
    - 在异步代码中，**绝不要**因为磁盘/网络 I/O 或 CPU 密集型任务而**阻塞事件循环**。

## 风格与规范 (Style & conventions)

- 使用描述性的名称；避免使用晦涩难懂的单行代码。
- 优先使用 `pathlib` 而非 `os.path`。
- 除非引入依赖项确实物超所值，否则优先使用标准库。
- 使用 json.dump 或 json.dumps 时，必须设置 ensure_ascii=False，以确保输出内容支持中文原样显示，提高可读性。
- 所有入口复用同一个 Core，而不是各做一套业务逻辑。

## 异步 (适用时)

- 不要在 `async def` 中运行阻塞式 I/O 或 CPU 密集型任务。
- 使用 `asyncio.to_thread(...)`（必要时使用 executor）将阻塞任务转移出去。
- 在可能的情况下，保持异步 API 从端到端的一致性。

## Git 与提交规范 (Git & Commit conventions)
- 每次代码改动后，必须提供建议的 commit 信息。
- Commit 信息应简洁、准确地描述本次修改内容。
- 优先使用 **Conventional Commits** 格式，结构分为：`类型(可选范围): 简短标题` + **空一行** + 多行详细描述。

### 基础类型标签
  - `feat: add xxx functionality`（新增功能）
  - `fix: resolve xxx issue`（修复问题）
  - `refactor: simplify xxx logic`（重构代码，无行为变更）
  - `docs: update documentation`（文档修改）
  - `test: add tests for xxx`（测试相关）
  - `perf: improve xxx performance`（性能优化）
  - `chore: build / dependency / tooling change`（工程构建、依赖、工具改动）

### 多行提交消息格式规范
> 标题行控制在 50 字符以内；标题与详情之间**必须保留一个空行**；详情每条改动以 `- ` 开头，一行一条改动点。