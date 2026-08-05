# Financial Intelligent Investment Research Agent (RAG + MCP)

Language: [中文](README.md) | English

## Table of Contents

- [Financial Intelligent Investment Research Agent (RAG + MCP)](#financial-intelligent-investment-research-agent-rag--mcp)
  - [Table of Contents](#table-of-contents)
  - [1. Capabilities](#1-capabilities)
  - [2. Models and Tech Stack](#2-models-and-tech-stack)
  - [3. Architecture](#3-architecture)
  - [4. Run Guide](#4-run-guide)
    - [4.1 Install dependencies](#41-install-dependencies)
    - [4.2 Configure environment](#42-configure-environment)
    - [4.3 Build index](#43-build-index)
    - [4.4 Common commands](#44-common-commands)
    - [4.5 Chat Memory](#45-chat-memory)
    - [4.6 Example outputs (real run excerpts)](#46-example-outputs-real-run-excerpts)
  - [5. Common Errors](#5-common-errors)
  - [6. Future Roadmap](#6-future-roadmap)
    - [6.1 MCP capability expansion](#61-mcp-capability-expansion)
    - [6.2 Vector retrieval scaling path](#62-vector-retrieval-scaling-path)
    - [6.3 Agent evolution](#63-agent-evolution)
    - [6.4 Memory mechanism evolution](#64-memory-mechanism-evolution)
  - [7. How to Build Your Own Knowledge Base (Not Limited to Banking)](#7-how-to-build-your-own-knowledge-base-not-limited-to-banking)
    - [7.1 Single-Industry Knowledge Base (Recommended Starting Point)](#71-single-industry-knowledge-base-recommended-starting-point)
    - [7.2 Multi-Industry Knowledge Base (Scaling Path)](#72-multi-industry-knowledge-base-scaling-path)
    - [7.3 Document Quality Criteria (Critical for Retrieval Quality)](#73-document-quality-criteria-critical-for-retrieval-quality)
    - [7.4 Recommended Build Workflow](#74-recommended-build-workflow)
    - [7.5 Acceptance Criteria for a High-Quality Knowledge Base](#75-acceptance-criteria-for-a-high-quality-knowledge-base)

This project is a localized intelligent research assistant for the banking domain. It combines report retrieval (RAG) and financial calculation (MCP) into one unified agent, covering the full flow from question understanding to retrieval, tool execution, and final conclusion.

## 1. Capabilities

The system provides five core capabilities:

1. Report-grounded Q&A (RAG)
- Retrieves relevant passages from local banking research reports.
- Produces traceable answers with sources.

2. Financial metric calculation (MCP)
- Static PE calculation: calc_pe
- Debt ratio calculation: calc_debt_ratio
- Unified JSON protocol with parameter validation and clear error handling.

3. Hybrid reasoning (RAG + MCP)
- Retrieves industry/policy context first, then runs quantitative calculation.
- Prefers extracting industry benchmarks from RAG text, reducing dependency on explicit numeric benchmarks in user questions.

4. Intelligent routing (Agent)
- Automatically routes user queries into: rag_only / mcp_only / rag_mcp / general
- End-to-end LLM-first:
  - LLM-first intent routing
  - LLM-first MCP tool and parameter parsing
  - Rule-based fallback when LLM output is unavailable or malformed

5. ReAct dynamic orchestration (unified Q&A entry)
- ask / chat / agent-ask all go through Agent orchestration, with ReAct enabled per query.
- ask / agent-ask are stateless single-turn calls; chat provides stateful multi-turn conversation within a session.
- ReAct runs as a Thought -> Action -> Observation loop, deciding whether to retrieve, compute, or finish.
- You can inspect structured step traces via --show-react-trace for better explainability.

6. Chat Memory
- SQLite stores ordered user/assistant messages and supplies them to the model as standard chat history.
- Confirmed preferences are shared across chat sessions with the same user ID and can be viewed, replaced, or deleted.
- Chat Memory does not depend on embeddings or Chroma; RAG knowledge retrieval remains separate.

## 2. Models and Tech Stack

Model roles:

- Embedding model (retrieval vectorization): BAAI/bge-small-zh-v1.5
- Generation model (answering and parsing): deepseek-chat (OpenAI-compatible API)

Core components:

- Vector database: Chroma (local persistence under data/chroma)
- Document parsing: PyPDFLoader (based on pypdf)
- OCR preprocessing: PyMuPDF + RapidOCR
- Text splitting: RecursiveCharacterTextSplitter
- Orchestration framework: LangChain
- Model client: ChatOpenAI
- Environment management: python-dotenv
- ReAct orchestrator: AgentRouter (unified multi-step reasoning and tool orchestration at the Agent layer)
- Chat Memory: SQLite (ordered session messages and user preference profiles)

Key .env variables:

- EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
- OPENAI_MODEL=deepseek-chat
- OPENAI_BASE_URL=https://api.deepseek.com/v1
- OPENAI_API_KEY=... (or DEEPSEEK_API_KEY=...)
- HF_ENDPOINT=https://hf-mirror.com
- HF_HUB_ETAG_TIMEOUT=60
- HF_HUB_DOWNLOAD_TIMEOUT=120
- ENABLE_OCR=true
- OCR_DPI=200
- MIN_PAGE_CHARS=20
- ENABLE_REACT=true (enable ReAct retrieval-reasoning loop)
- REACT_MAX_STEPS=3 (maximum ReAct iterations)
- MEMORY_DIR=./data/memory (SQLite chat memory directory)
- MEMORY_PROFILE_MIN_CONFIDENCE=0.7 (minimum confirmed profile confidence)
- CHAT_HISTORY_LIMIT=20 (recent messages loaded for each chat turn)

## 3. Architecture

The system follows a five-layer design:

1. Data layer
- Local report corpus: Knowledge/industry_bank, Knowledge/stock_bank, Knowledge/macro_policy

2. RAG retrieval layer
- PDF loading, OCR fallback, chunking, embedding, and similarity retrieval

3. MCP tools layer
- Standardized request/response protocol
- Tool registration and dispatching
- Validation and exception handling

4. Agent orchestration layer
- LLM-first intent classification
- ReAct loop driven: Thought -> Action -> Observation
- Unified action execution: rag_retrieve (retrieve) / mcp_call (compute) / finish (stop)
- Auto-routing into RAG, MCP, or hybrid reasoning, with structured answers and sources

5. Chat Memory layer
- SQLite session messages scoped by session ID
- User preference profiles shared across chat sessions with the same user ID
- Independent from the RAG vector store

## 4. Run Guide

### 4.1 Install dependencies

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 4.2 Configure environment

```powershell
Copy-Item .env.example .env
```

At minimum, configure EMBEDDING_MODEL, OPENAI_MODEL, OPENAI_BASE_URL, and OPENAI_API_KEY (or DEEPSEEK_API_KEY).

The current `.env.example` can be copied directly. Legacy scope/project memory variables may be omitted; their code defaults do not control Chat Memory.

### 4.3 Build index

```powershell
python -m src.main build-index
```

Quick smoke test (only a few PDFs):

```powershell
python -m src.main build-index --max-pdfs 1
```

### 4.4 Common commands

1. Full-feature Q&A (Unified Agent orchestration)

```powershell
python -m src.main ask --question "How did net interest margins trend for banks in 2024?"
python -m src.main ask --question "How did net interest margins trend for banks in 2024?" --show-react-trace
```

2. Full-feature interactive chat (Unified Agent orchestration)

```powershell
python -m src.main chat
python -m src.main chat --show-react-trace
python -m src.main chat --user-id u001 --session-id s001
```

3. MCP tools

```powershell
python -m src.main mcp-tools
python -m src.main mcp-call -p "{\"tool_name\":\"calc_pe\",\"params\":{\"price\":8.25,\"eps\":0.68}}"
python -m src.main mcp-nl -q "CITIC Bank price is 8.25 and EPS is 0.68, calculate PE"
```

4. Agent routing and end-to-end run

```powershell
python -m src.main agent-route -q "Industry average PE is 13.5x, CITIC Bank price is 8.25 and EPS is 0.68, evaluate valuation"
python -m src.main agent-ask -q "Please evaluate CITIC Bank valuation using research context. Price is 8.25 and EPS is 0.68"
python -m src.main agent-ask -q "What is provision coverage ratio for banks?"
python -m src.main agent-ask -q "What is provision coverage ratio for banks?" --show-react-trace
```

Note: `ask / chat / agent-ask` use the same orchestration path with ReAct enabled. Only `chat` persists conversation history; `ask / agent-ask` remain stateless.
`--show-react-trace` is optional. By default, fine-grained ReAct trace is hidden and only answer + route + sources are shown. Enabling this flag prints structured step-by-step traces with `Thought / Action / Observation`.

5. Regression tests

```powershell
python scripts/test_mcp_tools.py
python scripts/test_agent_router.py
python scripts/test_memory_store.py
```

### 4.5 Chat Memory

Chat Memory is enabled only for `chat` and has two parts:

- **Short-term memory**: ordered user/assistant messages scoped by `session-id`.
- **Long-term preferences**: explicit, high-confidence preferences scoped by `user-id` and reusable across sessions.

`ask / agent-ask` remain stateless.

1. Configure and run

```powershell
# .env
MEMORY_DIR=./data/memory
MEMORY_PROFILE_MIN_CONFIDENCE=0.7
CHAT_HISTORY_LIMIT=20

# Start or restore a stateful chat
python -m src.main chat --user-id u001 --session-id s001
```

2. Manage preferences

```powershell
python -m src.main profile-show --user-id u001
python -m src.main profile-set --user-id u001 --preference output_style=bullet-points --preference risk_preference=conservative
python -m src.main profile-delete --user-id u001
```

3. How it works

Chat Memory is stored in `MEMORY_DIR/chat_memory.sqlite3`. `chat_sessions` stores sessions, `chat_messages` stores ordered messages, and `user_profiles` stores preferences. Each turn:

1. Loads recent messages for the current session and the user profile from SQLite.
2. Persists the current user message, then supplies history to Agent/ReAct as standard chat messages.
3. Persists the assistant message after a successful answer; the next turn or restored session reloads from SQLite.

Messages are persisted per turn, not batched when chat exits. `CHAT_HISTORY_LIMIT=20` loads at most 20 messages, usually about 10 complete turns. Older messages remain in SQLite but are excluded from the current prompt.

4. Current limitations

- **Automatic Summary is not implemented yet**: `chat_sessions.summary` is only a reserved field. Messages outside the 20-message window are not summarized, so the model sees only the most recent 20 messages.
- Reading chat history uses only SQLite. Embeddings and Chroma are loaded only when the agent actually executes `rag_retrieve`.
- Preferences are stored and injected, but final formatting still depends on LLM instruction following.

### 4.6 Example outputs (real run excerpts)

The following excerpts are captured from real local runs for quick verification.

1. Session short-term memory

```powershell
python -m src.main chat --user-id readme-memory-test --session-id readme-memory-20260805
```

The first turn loaded 0 messages. The second turn loaded 2 messages and reused the previous `8.25 / 0.68 = 12.13x` example. After exiting and reopening the same session, the model loaded 10 messages and correctly identified the first question:

```text
=== Answer ===
Your first question in this session was: "CITIC Bank price is 8.25 and EPS is 0.68, calculate static PE."

=== Chat Memory ===
- Loaded messages: 10
- User profile loaded: No
- Message write: ok
```

This uses general chat history; there is no special "previous question" handler.

2. Cross-session user preferences

An explicit preference produced `profile write: updated`. A new session with the same user ID then showed:

```text
=== Chat Memory ===
- Loaded messages: 0
- User profile loaded: Yes
- Message write: ok
```

The preference for parenthetical explanations appeared in the answer. Bullet-point formatting was not consistent, confirming that preference injection is not the same as enforced formatting.

3. ask command (full-feature orchestration, auto-routes to MCP)

Command:

```powershell
python -m src.main ask --question "CITIC Bank price is 8.25 and EPS is 0.68, calculate static PE"
```

Output (excerpt):

```text
=== Route ===
mcp_only

=== Answer ===
Static PE of CITIC Bank = Price ÷ EPS = 8.25 ÷ 0.68 ≈ 12.13x.

=== Trace ===
- LLM recognized this as a pure numeric calculation request based on price and EPS, without requiring retrieval.
- Agent-level ReAct is enabled (Thought -> Action -> Observation).
```

4. ask command (show structured ReAct trace)

Command:

```powershell
python -m src.main ask --question "What is provision coverage ratio for banks?" --show-react-trace
```

Output (excerpt):

```text
=== Route ===
general

=== Trace ===
- LLM recognized this as a financial concept question, without mandatory retrieval or numeric computation.
- Agent-level ReAct is enabled (Thought -> Action -> Observation).

=== Detailed ReAct Trace ===
[Step 1]
Thought:
  This is a definition-style question; direct completion is sufficient.
Action:
  finish
Observation:
  Loop ended and final answer returned.
```

5. agent-ask command (hybrid RAG + MCP)

Command:

```powershell
python -m src.main agent-ask -q "Industry average PE is 13.5x, CITIC Bank price is 8.25 and EPS is 0.68, judge valuation" --show-react-trace
```

Output (excerpt):

```text
=== Route ===
rag_mcp

=== Trace ===
- LLM recognized this as a hybrid request: industry benchmark retrieval + PE computation.
- Agent-level ReAct is enabled (Thought -> Action -> Observation).

=== Detailed ReAct Trace ===
[Step 1] Action: rag_retrieve
[Step 2] Action: mcp_call
[Step 3] Action: rag_retrieve
```

## 5. Common Errors

1. Cannot connect to huggingface.co / embedding download fails
- Symptom: network errors during index build.
- Fix: set HF_ENDPOINT=https://hf-mirror.com in .env and retry in a new terminal.

2. No valid text extracted from PDF
- Symptom: empty text/content warnings during indexing.
- Fix: set ENABLE_OCR=true, increase OCR_DPI to 250 or 300, or convert scanned PDFs to text-searchable PDFs first.

3. Vector index not found
- Symptom: ask/chat/agent-ask prompts to build index first.
- Fix: run python -m src.main build-index.

4. LLM auth or endpoint errors
- Symptom: API errors in agent-route, agent-ask, or mcp-nl.
- Fix: verify OPENAI_BASE_URL, OPENAI_MODEL, OPENAI_API_KEY (or DEEPSEEK_API_KEY).
- Note: fallback rules exist, but complete LLM outage still limits RAG answering and LLM parsing quality.

5. Chroma telemetry warnings (for example, capture argument errors)
- Symptom: telemetry warnings in terminal output.
- Fix: usually non-blocking; can be ignored temporarily. For a permanent fix, lock or upgrade related dependency versions.

## 6. Future Roadmap

### 6.1 MCP capability expansion

Current MCP tools include calc_pe and calc_debt_ratio. Future expansion can build a banking research metrics toolkit:

1. Valuation tools
- Dividend yield
- PB valuation and PB-ROE comparison
- PEG valuation

2. Asset quality and risk tools
- NPL ratio
- Provision coverage ratio
- CET1 ratio

3. Profitability tools
- ROE/ROA and peer percentile comparison
- NIM trend analysis
- Cost-to-income ratio (CIR)

4. Joint analysis tools
- Auto-generate benchmark cards (industry average vs stock metrics)
- Multi-period YoY/QoQ explanation
- Structured RAG-linked conclusions (conclusion, evidence, risk)

Recommended implementation:
- Keep the current JSON protocol and register new tools with explicit parameter schema.
- Add boundary checks and unit tests for each new tool to prevent metric-definition drift.

### 6.2 Vector retrieval scaling path

Chroma local persistence is suitable for demos and small-to-medium datasets. As scale grows (multi-industry, multi-year, million-level chunks), a staged upgrade is recommended:

1. Medium scale (100k to 1M chunks)
- Preferred: Milvus or Qdrant
- Benefits: high-performance ANN, stronger filtering, containerized deployment, easier horizontal scaling

2. Large-scale production (million+ chunks, multi-user concurrency)
- Options: Elasticsearch/OpenSearch (hybrid vector + keyword retrieval)
- Cloud-native options: Pinecone, Weaviate Cloud, Azure AI Search
- Benefits: managed operations, elastic scaling, better monitoring and access control

3. Recommended migration strategy
- Step 1: keep retriever interface abstraction and add backend adapters.
- Step 2: enable hybrid retrieval (vector recall + BM25 + reranker).
- Step 3: add caching and evaluation metrics (hit rate, answer quality, latency, cost).

### 6.3 Agent evolution

1. Routing and tool orchestration
- Upgrade from LLM-first to a three-stage policy: LLM decision + rule validation + confidence thresholds.
- Add secondary checks for key fields (units, reporting-period consistency).

2. Explainability
- Add calculation-method notes, data timestamps, and evidence snippets.
- Add uncertainty and coverage warnings for hybrid reasoning outputs.

3. Engineering and operations
- Add end-to-end regression suites and benchmark sets.
- Add logs and tracing for diagnosing retrieval misses and MCP parameter issues.
- Progressively support API service deployment and multi-session concurrency.

### 6.4 Memory mechanism evolution

The current Chat Memory implementation provides SQLite session messages and cross-session user preferences. The RAG knowledge base remains separate and is not conversation memory.

Planned work:

1. **Session Summary**: incrementally summarize messages outside the recent window and build prompts from summary + recent raw messages.
2. **Token budget**: replace the fixed message count with model-aware context-window trimming.
3. **Preference governance**: add field-level sources, version history, and stronger formatting controls.

## 7. How to Build Your Own Knowledge Base (Not Limited to Banking)

This project currently uses banking reports as the default example, but the same method works for both single-industry and multi-industry corpora. The key principles are:

1. Define use cases first, then collect documents.
2. Prioritize quality before volume.
3. Make single-industry retrieval stable first, then scale to multi-industry.

### 7.1 Single-Industry Knowledge Base (Recommended Starting Point)

Best for course projects, demos, and focused research in one sector (for example banking, semiconductors, healthcare).

Recommended corpus size:

- Starter set: 20 to 30 PDFs
- Production-ready baseline: 40 to 80 PDFs

Recommended document mix and ratio (single industry):

1. Industry deep-dive reports: 40% to 50%
- Focus: sector drivers, valuation center, policy impact, competitive landscape.

2. Leading-stock deep-dive reports: 30% to 40%
- Focus: business model, earnings forecast, valuation methodology, key risks.

3. Macro and policy special reports: 10% to 20%
- Focus: interest rates, regulation, fiscal/monetary context, industrial policy.

4. Earnings review and financial commentary: 10% to 15%
- Focus: quarter-over-quarter changes, key metric movement, management guidance.

Note: slight ratio variation is fine, but avoid an extreme mix such as only short stock notes or only macro reports.

### 7.2 Multi-Industry Knowledge Base (Scaling Path)

Best for sector comparison, asset allocation, and cross-industry research.

Recommended directory strategy:

- Create one subfolder per sector, for example:
  - Knowledge/industry_bank
  - Knowledge/industry_semiconductor
  - Knowledge/industry_medicine
  - Knowledge/macro_policy

Recommended document mix and ratio (multi-industry):

1. Sector deep-dive reports: 35% to 45%
2. Sector leading-stock deep dives: 30% to 35%
3. Cross-sector strategy/allocation reports: 10% to 15%
4. Macro/policy reports: 10% to 15%
5. Earnings commentary and event reviews: 5% to 10%

Additional recommendations for multi-industry corpora:

1. Add sector labels for each document (in filename or metadata).
2. Route by sector first, then run vector retrieval.
3. Avoid mixing metric definitions across sectors (valuation and profit-calculation conventions can differ).

### 7.3 Document Quality Criteria (Critical for Retrieval Quality)

Prefer documents that are:

1. Long-form and recent (typically within the last 2 to 3 years, ideally over 10 pages).
2. Rich in charts/tables with clear metric definitions and cited data sources.
3. Explicit about valuation assumptions and methodology.

Remove or downweight documents that are:

1. Very short notes (1 to 3 pages), news flashes, or quick comments.
2. Pure scanned images with poor OCR quality.
3. Too old with outdated logic.
4. Highly repetitive reposts with little incremental information.

### 7.4 Recommended Build Workflow

1. Build folder structure and place PDFs by category (single-industry or multi-industry).
2. Run a small indexing smoke test:

```powershell
python -m src.main build-index --max-pdfs 10
```

3. Validate with 10 to 20 real user questions:
- Sector trend questions
- Stock fundamental questions
- Policy-impact questions
- Metric comparison questions

4. Check whether retrieval is relevant, citable, and explainable.
5. Then scale up to the full corpus.

### 7.5 Acceptance Criteria for a High-Quality Knowledge Base

Use these five checks:

1. Coverage: for mainstream questions, at least 2 to 3 evidence passages are retrieved.
2. Traceability: key conclusions can be mapped to concrete source passages.
3. Stability: repeated semantically similar questions return consistent conclusions.
4. Low noise: irrelevant retrieval rate remains low.
5. Scalability: adding new industries does not significantly degrade existing domain quality.
