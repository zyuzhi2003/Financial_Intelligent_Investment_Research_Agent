# 这个文件用于说明纯RAG模块的目标、运行步骤与测试方式。

## 金融RAG模块（阶段二）

本目录实现了一个可独立运行的纯RAG问答系统，能力包括：

- 扫描 `Knowledge/` 下银行业研报PDF
- 自动切片并构建本地Chroma向量库
- 相似度检索并拼接上下文
- 约束式问答（必须基于检索内容，附来源）

## 模型与技术栈说明

本项目同时使用了“Embedding模型”和“生成模型”，二者分工不同：

- Embedding模型（用于检索）：`BAAI/bge-small-zh-v1.5`
	- 作用：把研报切片和用户问题向量化，用于相似度检索
	- 使用阶段：`build-index` 建库 + `ask/chat` 检索

- 生成模型（用于回答）：`deepseek-chat`（DeepSeek Flash兼容接口）
	- 作用：基于检索到的上下文生成最终答案
	- 使用阶段：`ask/chat` 回答生成

核心技术组件如下：

- 向量数据库：`Chroma`（本地落盘，目录为 `data/chroma`）
- 文档解析：`PyPDFLoader`（基于pypdf）
- OCR预处理：`PyMuPDF + RapidOCR`（用于扫描版PDF）
- 文本切分：`RecursiveCharacterTextSplitter`
- RAG编排：`LangChain`
- 模型接入：`ChatOpenAI`（通过OpenAI兼容协议调用DeepSeek）
- 环境变量管理：`python-dotenv`

默认关键配置（可在 `.env` 覆盖）：

- `EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5`
- `OPENAI_MODEL=deepseek-chat`
- `OPENAI_BASE_URL=https://api.deepseek.com/v1`
- `HF_ENDPOINT=https://hf-mirror.com`
- `ENABLE_OCR=true`
- `OCR_DPI=200`
- `MIN_PAGE_CHARS=20`

## 1. 安装依赖

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 2. 配置环境变量

```powershell
Copy-Item .env.example .env
```

把 `.env` 中的 `DEEPSEEK_API_KEY` 改成你的Key（默认已配置为DeepSeek Flash兼容接口）。

## 3. 构建向量库

```powershell
python -m src.main build-index
```

说明：若PDF为扫描件或无法直接复制文本，程序会在 `ENABLE_OCR=true` 时自动对空白页做OCR补抽。

快速测试（只跑1个PDF，便于验证流程和缩短时间）：

```powershell
python -m src.main build-index --max-pdfs 1
```

## 4. 执行问答

```powershell
python -m src.main ask --question "2026银行手续费收入利不利好？"
```

## 5. 交互模式

```powershell
python -m src.main chat
```

输入 `exit` 可退出。

## 6. 常见报错排查

如果在 `build-index` 阶段出现无法连接 `huggingface.co` 的报错：

1. 确认 `.env` 中已设置 `HF_ENDPOINT=https://hf-mirror.com`
2. 重新打开终端后再执行构建命令
3. 如仍失败，可在网络可用环境先下载 `BAAI/bge-small-zh-v1.5` 到本机缓存后离线运行

如果在 `build-index` 阶段遇到“未从PDF中提取到有效文本”：

1. 确认 `.env` 中 `ENABLE_OCR=true`
2. 将 `OCR_DPI` 调到 `250` 或 `300` 后重试
3. 若仍失败，说明PDF清晰度过低或版式过于复杂，建议先转换为可复制文本PDF
