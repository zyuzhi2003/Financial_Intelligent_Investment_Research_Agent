# 金融智能投研 Agent（RAG + MCP）

语言选择：中文 | [English](README_EN.md)

## 目录

- [金融智能投研 Agent（RAG + MCP）](#金融智能投研-agentrag--mcp)
  - [目录](#目录)
  - [1. 能力介绍](#1-能力介绍)
  - [2. 模型与技术栈说明](#2-模型与技术栈说明)
  - [3. 架构介绍](#3-架构介绍)
  - [4. 运行流程](#4-运行流程)
    - [4.1 安装依赖](#41-安装依赖)
    - [4.2 配置环境变量](#42-配置环境变量)
    - [4.3 构建索引](#43-构建索引)
    - [4.4 运行方式](#44-运行方式)
    - [4.5 Memory使用（短期/用户/项目）](#45-memory使用短期用户项目)
    - [4.6 运行结果示例（实测输出摘录）](#46-运行结果示例实测输出摘录)
  - [5. 常见报错](#5-常见报错)
  - [6. 未来展望](#6-未来展望)
    - [6.1 MCP工具能力扩展](#61-mcp工具能力扩展)
    - [6.2 向量检索架构升级路径](#62-向量检索架构升级路径)
    - [6.3 Agent能力演进](#63-agent能力演进)
    - [6.4 Memory机制演进](#64-memory机制演进)
  - [7. 如何构建自己的 Knowledge 库（不限银行业）](#7-如何构建自己的-knowledge-库不限银行业)
    - [7.1 单行业知识库（推荐起步方式）](#71-单行业知识库推荐起步方式)
    - [7.2 多行业知识库（扩展方式）](#72-多行业知识库扩展方式)
    - [7.3 资料筛选标准（高质量知识库关键）](#73-资料筛选标准高质量知识库关键)
    - [7.4 推荐构建流程](#74-推荐构建流程)
    - [7.5 一个“好知识库”的验收标准](#75-一个好知识库的验收标准)

本项目是一个面向银行业场景的本地化智能投研问答系统，将“研报检索”与“财务测算”合并到一个统一Agent中，支持从问题理解、资料检索、工具计算到结论输出的完整链路。

## 1. 能力介绍

系统提供六类核心能力：

1. 研报检索问答（RAG）
- 从本地银行业研报中检索相关片段并生成回答。
- 回答强调可追溯，输出参考来源。

2. 财务指标计算（MCP）
- 支持静态PE计算：`calc_pe`
- 支持资产负债率计算：`calc_debt_ratio`
- 统一JSON协议，支持参数校验与错误返回。

3. 检索+计算联合推理（RAG + MCP）
- 先检索行业/政策语境，再执行量化计算。
- 优先从RAG文本抽取行业基准值做对比，减少对问题中显式数字的依赖。

4. 智能分流调度（Agent）
- 自动识别问题属于：`rag_only` / `mcp_only` / `rag_mcp` / `general`
- 全链路 LLM-first：
  - 路由意图优先由LLM判定
  - MCP工具与参数优先由LLM解析
  - LLM异常时自动回退规则，保证可用性

5. ReAct动态编排（统一问答入口）
- `ask / chat / agent-ask` 统一走Agent调度，并在每轮问答中启用ReAct机制。
- ReAct按 `Thought -> Action -> Observation` 闭环执行，动态决定下一步是检索、计算还是结束回答。
- 支持通过 `--show-react-trace` 查看结构化步骤轨迹，提升过程可解释性。

6. 分层Memory（已实现）
- 会话记忆（session）：同一会话内记住上下文。
- 用户记忆（user）：沉淀用户偏好（风险偏好、投资期限、输出风格等）。
- 项目记忆（project）：沉淀可复用的项目结论。
- 支持TTL清理、手动查询/清理、语义检索（SQLite主存 + Chroma语义层）。

## 2. 模型与技术栈说明

模型分工：

- Embedding模型（检索向量化）：`BAAI/bge-small-zh-v1.5` （适用中文语境，若是英文建议换成`EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2`）
- 生成模型（问答与解析）：`deepseek-chat`（OpenAI兼容接口）

核心组件：

- 向量数据库：`Chroma`（本地持久化目录 `data/chroma`）
- 文档解析：`PyPDFLoader`（基于pypdf）
- OCR预处理：`PyMuPDF + RapidOCR`
- 文本切分：`RecursiveCharacterTextSplitter`
- 编排框架：`LangChain`
- 模型接入：`ChatOpenAI`
- 环境管理：`python-dotenv`
- ReAct编排器：`AgentRouter`（在Agent层统一执行多步推理与工具调度）
- Memory存储：`SQLite + Chroma`（SQLite事实主存，Chroma语义召回）

关键环境变量（`.env`）：

- `EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5`
- `OPENAI_MODEL=deepseek-chat`
- `OPENAI_BASE_URL=https://api.deepseek.com/v1`
- `OPENAI_API_KEY=...`（或 `DEEPSEEK_API_KEY=...`）
- `HF_ENDPOINT=https://hf-mirror.com`
- `ENABLE_OCR=true`
- `OCR_DPI=200`
- `MIN_PAGE_CHARS=20`
- `ENABLE_REACT=true`（是否启用ReAct检索-推理循环）
- `REACT_MAX_STEPS=3`（ReAct最多迭代步数）
- `MEMORY_ENABLED=true`（是否启用Memory）
- `MEMORY_BACKEND=hybrid`（`sqlite` / `hybrid`）
- `MEMORY_DIR=./data/memory`
- `MEMORY_TOP_K=3`
- `MEMORY_TTL_DAYS=30`
- `MEMORY_COLLECTION=agent_memory`
- `MEMORY_PROFILE_MIN_CONFIDENCE=0.7`（用户画像确认阈值）

## 3. 架构介绍

系统采用五层结构：

1. 数据层
- 本地研报知识库：`Knowledge/industry_bank`、`Knowledge/stock_bank`、`Knowledge/macro_policy`

2. RAG检索层
- PDF加载、OCR补抽、文本切片、向量化入库、相似度召回

3. MCP工具层
- 标准化请求/响应协议
- 财务工具注册与调度
- 参数校验与异常处理

4. Agent调度层
- LLM-first意图识别
- ReAct循环驱动：`Thought -> Action -> Observation`
- Action统一执行：`rag_retrieve`（检索）/ `mcp_call`（计算）/ `finish`（结束）
- 自动分流到RAG、MCP或联合推理，并输出结构化答案与参考来源

5. Memory层
- 分层记忆：`session / user / project`
- 写入策略：会话写回、用户画像分层写回（候选/确认）、项目结论沉淀
- 治理策略：TTL过期清理、按scope/user/session手动清理

## 4. 运行流程

### 4.1 安装依赖

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 4.2 配置环境变量

```powershell
Copy-Item .env.example .env
```

至少配置：`EMBEDDING_MODEL`、`OPENAI_MODEL`、`OPENAI_BASE_URL`、`OPENAI_API_KEY`（或`DEEPSEEK_API_KEY`）。

### 4.3 构建索引

```powershell
python -m src.main build-index
```

快速验证（仅处理少量PDF）：

```powershell
python -m src.main build-index --max-pdfs 1
```

### 4.4 运行方式

1. 全功能问答（Agent统一调度）

```powershell
python -m src.main ask --question "2026银行手续费收入利不利好？"
python -m src.main ask --question "2026银行手续费收入利不利好？" --show-react-trace
python -m src.main ask --question "中信银行现价8.25元，每股收益EPS0.68，计算静态PE" --user-id u001 --session-id s001
```

2. 全功能交互问答（Agent统一调度）

```powershell
python -m src.main chat
python -m src.main chat --show-react-trace
python -m src.main chat --user-id u001
```

3. MCP工具（查看/调用）

```powershell
python -m src.main mcp-tools
python -m src.main mcp-call -p "{\"tool_name\":\"calc_pe\",\"params\":{\"price\":8.25,\"eps\":0.68}}"
python -m src.main mcp-nl -q "中信银行现价8.25元，每股收益EPS0.68，计算静态PE"
```

4. Agent路由与全链路调度

```powershell
python -m src.main agent-route -q "股份行行业平均PE13.5倍，中信银行股价8.25元，EPS0.68，判断估值高低"
python -m src.main agent-ask -q "请结合研报判断中信银行当前估值高低，已知股价8.25元、EPS0.68"
python -m src.main agent-ask -q "什么是银行拨备覆盖率？"
python -m src.main agent-ask -q "什么是银行拨备覆盖率？" --show-react-trace
```

说明：`ask / chat / agent-ask` 现在统一走全功能调度（自动判断RAG/MCP/联合推理），并启用ReAct机制。
`--show-react-trace` 为可选开关，默认不展示细粒度轨迹，仅输出答案、执行路径与来源；开启后会按 Step 结构展示 `Thought / Action / Observation`。

5. 回归测试

```powershell
python scripts/test_mcp_tools.py
python scripts/test_agent_router.py
python scripts/test_memory_store.py
```

### 4.5 Memory使用（短期/用户/项目）

1. 开启Memory

```powershell
# .env
MEMORY_ENABLED=true
MEMORY_BACKEND=hybrid
MEMORY_DIR=./data/memory
MEMORY_TOP_K=3
MEMORY_TTL_DAYS=30
MEMORY_COLLECTION=agent_memory
MEMORY_PROFILE_MIN_CONFIDENCE=0.7
```

2. 带身份问答（触发session/user记忆）

```powershell
python -m src.main agent-ask -q "我风险偏好保守，投资期限3到5年，请以后要点式回答" --user-id u001 --session-id s001
```

3. 查询记忆

```powershell
python -m src.main memory-list --scope session --user-id u001 --session-id s001 --limit 5
python -m src.main memory-list --scope user --user-id u001 --limit 5
python -m src.main memory-list --scope project --limit 5
python -m src.main memory-search -q "静态PE" --scope session --user-id u001 --session-id s001 --top-k 3
```

4. 清理记忆

```powershell
python -m src.main memory-prune
python -m src.main memory-clear --scope session --user-id u001 --session-id s001
```

### 4.6 运行结果示例（实测输出摘录）

以下结果来自本项目在本地环境的实际运行，便于快速对照。

1. ask命令（全功能调度，自动走MCP）

命令：

```powershell
python -m src.main ask --question "中信银行现价8.25元，每股收益EPS0.68，计算静态PE"
```

输出（节选）：

```text
=== 执行路径 ===
mcp_only

=== 回答 ===
中信银行静态PE = 股价 ÷ EPS = 8.25 ÷ 0.68 ≈ 12.13倍。

=== 调度轨迹 ===
- LLM识别：用户仅要求根据给定价格和EPS计算静态PE，属于纯数值计算，无需资料查询。
- 已启用Agent级ReAct编排（Thought -> Action -> Observation）。
```

2. ask命令（展示结构化ReAct轨迹）

命令：

```powershell
python -m src.main ask --question "什么是银行拨备覆盖率？" --show-react-trace
```

输出（节选）：

```text
=== 执行路径 ===
general

=== 回答 ===
银行拨备覆盖率是衡量银行对贷款损失准备计提充足程度的重要指标...

=== 调度轨迹 ===
- LLM识别：这是关于金融概念的常识性问答，不涉及具体资料查询或数值计算。
- 已启用Agent级ReAct编排（Thought -> Action -> Observation）。

=== ReAct 详细轨迹 ===
[Step 1]
Thought:
  用户询问银行拨备覆盖率的定义，这是一个基础概念问题，不需要检索或计算，直接基于常识回答即可。
Action:
  finish
Observation:
  结束循环并产出最终回答。
```

3. agent-ask命令（RAG+MCP联合场景）

命令：

```powershell
python -m src.main agent-ask -q "股份行行业平均PE13.5倍，中信银行股价8.25元，EPS0.68，判断估值高低" --show-react-trace
```

输出（节选）：

```text
=== 执行路径 ===
rag_mcp

=== 调度轨迹 ===
- LLM识别：用户要求计算PE并结合行业均值判断，属于资料查询+数值测算联合场景。
- 已启用Agent级ReAct编排（Thought -> Action -> Observation）。

=== ReAct 详细轨迹 ===
[Step 1] Action: rag_retrieve
[Step 2] Action: mcp_call
[Step 3] Action: rag_retrieve
```

## 5. 常见报错

1. 无法连接 huggingface.co / 下载Embedding模型失败
- 现象：建库时报网络连接错误。
- 处理：确认 `.env` 中 `HF_ENDPOINT=https://hf-mirror.com`；重新打开终端后重试。

2. 未从PDF提取到有效文本
- 现象：建库提示空文本或文档无有效内容。
- 处理：确认 `ENABLE_OCR=true`；将 `OCR_DPI` 调到 `250` 或 `300`；必要时先将PDF转为可复制文本。

3. 未检测到向量索引
- 现象：执行 `ask/chat/agent-ask` 提示先构建索引。
- 处理：先运行 `python -m src.main build-index`。

4. LLM鉴权或模型接口错误
- 现象：`agent-route`、`agent-ask` 或 `mcp-nl` 报API相关错误。
- 处理：检查 `OPENAI_BASE_URL`、`OPENAI_MODEL`、`OPENAI_API_KEY`（或 `DEEPSEEK_API_KEY`）是否正确。
- 说明：系统具备规则兜底，但若生成模型完全不可用，RAG回答与LLM解析能力会受限。

5. Chroma telemetry 警告（如 capture 参数错误）
- 现象：终端出现 telemetry warning。
- 处理：通常不影响主流程，可先忽略；若需彻底消除，可后续单独升级/锁定相关依赖版本。

## 6. 未来展望

### 6.1 MCP工具能力扩展

当前MCP工具已覆盖`calc_pe`与`calc_debt_ratio`。后续可扩展为“银行投研指标工具箱”：

1. 估值类工具
- 股息率（Dividend Yield）
- PB估值与PB-ROE对比
- PEG估值（适用于有增长预测的场景）

2. 资产质量与风险类工具
- 不良贷款率（NPL）
- 拨备覆盖率
- 核心一级资本充足率

3. 盈利能力类工具
- ROE/ROA计算与同业分位对比
- 净息差（NIM）趋势测算
- 成本收入比（CIR）

4. 联动分析工具
- 自动生成“行业均值 vs 个股指标”对标卡片
- 支持多期财报输入，生成同比/环比变化解释
- 与RAG联动输出结构化结论（结论、依据、风险点）

建议实现方式：
- 延续当前统一JSON协议，新增工具仅需注册函数与参数Schema。
- 为每个工具补充输入约束、边界检查和单测，避免计算口径漂移。

### 6.2 向量检索架构升级路径

当前使用`Chroma`本地持久化，适合Demo与中小规模数据。若数据量显著增长（如扩展到多行业、多年份、百万级切片），建议分阶段升级：

1. 中规模（约10万~100万切片）
- 优先方案：`Milvus` 或 `Qdrant`
- 优势：高性能ANN检索、过滤能力更强、可容器化部署、易于横向扩展。

2. 大规模与生产化（百万级以上，多用户并发）
- 可选方案：`Elasticsearch/OpenSearch`（向量+关键词混合检索）
- 云原生方案：`Pinecone`、`Weaviate Cloud`、`Azure AI Search`
- 优势：托管能力、弹性扩缩容、监控与权限管理更完善。

3. 推荐演进策略
- 第一步：保留现有Chroma接口抽象，先做检索层适配器（统一Retriever接口）。
- 第二步：上线混合检索（向量召回 + BM25关键词召回 + 重排模型）。
- 第三步：增加缓存与评估体系（命中率、回答准确率、延迟、成本）。

### 6.3 Agent能力演进

1. 路由与工具调度
- 从当前LLM-first升级为“LLM判定 + 规则校验 + 置信度阈值”三段式策略。
- 对关键问题启用二次确认（例如单位识别、财报口径一致性检查）。

2. 结果可解释性
- 输出中增加“计算口径说明”“数据时间戳”“引用证据片段”。
- 对联合推理结果增加风险提示（模型不确定性、样本覆盖不足）。

3. 工程化与运维
- 增加端到端回归集与基准测试集。
- 引入调用日志与链路追踪，定位RAG召回偏差与MCP参数错误。
- 逐步支持API服务化部署与多会话并发调用。

### 6.4 Memory机制演进

当前版本已落地分层Memory机制，并完成第二阶段第一批稳定性增强。

已实现：

1. 分层记忆
- Session Memory：保存当前会话上下文。
- User Memory：保存用户偏好画像。
- Project Memory：保存项目级结论。

2. 存储与治理
- 存储架构：SQLite主存 + Chroma语义检索。
- 默认TTL：30天，可通过命令手动清理。

3. 第二阶段第一批（已实现）
- 路由稳定化：对 `mcp_only / rag_mcp` 增加硬规则纠偏，降低同类问题漂移。
- 用户画像写回分层：低置信度不覆盖，中置信度写“候选画像”，高置信度写“确认画像”，并与历史画像合并。

后续建议：

1. 注入预算与优先级
- 对 session/user/project 记忆做预算分配与排序，降低噪声注入概率。

2. 候选画像转正策略
- 增加多轮一致性检查，避免单轮误判。

3. 规则参数化
- 将路由纠偏阈值与关键词配置化，支持按行业场景调优。

## 7. 如何构建自己的 Knowledge 库（不限银行业）

本项目当前示例以银行业为主，但同样支持单行业或多行业知识库。核心原则是：

1. 先定义场景，再收集资料。
2. 先保质量，再扩数量。
3. 先单行业跑通，再扩多行业。

### 7.1 单行业知识库（推荐起步方式）

适用场景：课程项目、Demo展示、单赛道投研（如银行、半导体、医药）。

推荐规模：

- 起步版本：20-30 份PDF
- 可用版本：40-80 份PDF

推荐资料类型与占比（单行业）：

1. 行业深度报告：40%-50%
- 内容重点：行业驱动因素、估值中枢、政策影响、竞争格局。

2. 龙头个股深度报告：30%-40%
- 内容重点：商业模式、盈利预测、估值方法、核心风险。

3. 宏观与政策专题：10%-20%
- 内容重点：利率、监管政策、财政货币环境、产业政策。

4. 财报解读与业绩综述：10%-15%
- 内容重点：季度变化、关键指标变动、管理层指引。

说明：上述占比可有轻微浮动，但不建议“全部是个股快评”或“全部是宏观报告”。

### 7.2 多行业知识库（扩展方式）

适用场景：做行业比较、资产配置、跨赛道研究。

推荐组织方式：

- 按行业建子目录，例如：
  - Knowledge/industry_bank
  - Knowledge/industry_semiconductor
  - Knowledge/industry_medicine
  - Knowledge/macro_policy

推荐资料类型与占比（多行业）：

1. 各行业深度报告：35%-45%
2. 各行业龙头个股深度：30%-35%
3. 跨行业策略与配置报告：10%-15%
4. 宏观/政策专题：10%-15%
5. 财报解读与事件复盘：5%-10%

多行业额外建议：

1. 为每个文档打行业标签（可在文件名或metadata中体现）。
2. 检索时优先做行业路由，再做向量召回。
3. 避免把同名指标跨行业混用（如估值口径、利润口径差异）。

### 7.3 资料筛选标准（高质量知识库关键）

建议优先保留以下资料：

1. 近 2-3 年的中长篇深度报告（优先10页以上）。
2. 含清晰图表、口径说明、数据来源说明的报告。
3. 含估值方法和假设条件的个股研究。

建议剔除以下资料：

1. 1-3 页短评、快讯类内容。
2. 纯图片扫描且OCR效果差的PDF。
3. 年份过旧且逻辑已失效的报告。
4. 大量重复转载、几乎无新增信息的文档。

### 7.4 推荐构建流程

1. 先建目录并分类放置PDF（单行业或多行业）。
2. 小规模试跑索引：

```powershell
python -m src.main build-index --max-pdfs 10
```

3. 用10-20个真实问题抽样验证：
- 行业趋势问题
- 个股基本面问题
- 政策影响问题
- 指标对比问题

4. 观察召回结果是否“相关、可引用、可解释”。
5. 再逐步扩容到完整知识库。

### 7.5 一个“好知识库”的验收标准

可以用下面5条快速判断：

1. 问题命中率：主流问题都能召回至少2-3段有效证据。
2. 引用可追溯：回答中的关键结论都能定位到具体文档片段。
3. 跨问题稳定：同类问题多次提问，结论一致性高。
4. 低噪声：无关文档召回比例低。
5. 可扩展：新增行业后，不明显破坏原有行业问答质量。
