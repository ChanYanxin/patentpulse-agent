# PatentPulse-Agent / TechIntel · MCP + 专利科技情报工程 MVP

**专利/科技情报分析增强版**，延续原 TechIntel RAG 框架；本项目面向研发人员的专业技术情报辅助工具：对研究资料检索并返回出处，提供文档工具、会话笔记、API、MCP Server、可复现评测。用于 Agent / LLM 应用算法工程师岗位展示。

> **真实性边界**：`corpus/demo` 内 **12 篇材料是人工合成教学笔记**；`corpus/patents_demo` 内 **4 篇是虚构专利记录**，不是论文、专利、业务数据或经过实证的结果。当前经过自动化测试的核心能力是本地 TF-IDF 混合检索、离线证据预览、引用 ID、SQLite memory 和 FastAPI；LLM 远程推理、MCP 客户端和 Docker runtime 要在你自己的环境中连接验证。`corpus/source_notes` 另有 **4 篇真实出版物的人工转述摘要**，`corpus/patents_source_notes` 还有 **3 篇真实专利出版物的人工转述摘要**（包含可核验 URL，不是论文全文）。**没有生产上线记录。**


## ⭐ 2026-10-16 High-Match v2：可演示增强版

在原 29 项测试的基础上，本版新增**36 项累计自动化测试全通过**、可浏览器操作的 `/demo` 页面、英文 Claims 数值列表与范围依赖解析、三份**真实专利出版物的人工转述摘要**及独立评测脚本。

启动 FastAPI 后可直接在 **http://127.0.0.1:8000/demo** 检索真实专利摘要、解析 Claim、生成术语覆盖对比图；`/docs` 仍可调试 API。

**真实 ID 语料只包含可追溯的人工摘要，不是完整专利数据：**
- [US11433555B2](https://patents.google.com/patent/US11433555B2/en)：机器人夹爪分层触觉感知
- [US20240243351A1](https://patents.google.com/patent/US20240243351A1/en)：通过材料结构设计缓解湿度影响的硫化物电解质
- [US20250140912A1](https://patents.google.com/patent/US20250140912A1/en)：掺杂型硫化物电解质的抗湿与离子传导

```bash
python -m techintel.real_patent_eval --save results/real_patent_notes_metrics.json
# Curated notes retrieval (new, no API key required):
# POST /patents/search
# {"query":"robotic gripper fingertip tactile sensing","source":"curated","retrieval":"bm25","k":3}
```

在 3 份人工转述摘要、10 个**作者自己编写**的可回答问题与 3 个无答案问题上，三种检索方法均 Hit@3 10/10；原始检索对知识库外问题均 0/3 拒答，经同一批问题上设计的“两项非泛化检索词重叠”启发式过滤后均 3/3 正确拒答。**这个前后对照不属于独立验证；有显著调参泄漏和短文档/简单查询局限，不能写成真实专利业务准确率。** 详见 `results/real_patent_notes_metrics.json` 和 `docs/CANDIDATE_24H_EXECUTION_CN.md`。

新增 `retrieval_enhanced.py` 包含 **BM25 + 真实 embedding 向量相似度 + RRF + 可选 CrossEncoder rerank** 的程序接口；依赖注入测试已通过，但未接入线上模型进行真实效果测试。用 OpenAI embeddings（需 Key 和费用）或本地 BGE-M3 + reranker（模型体积较大）：

```bash
python -m techintel.retrieval_enhanced 'robot gripper tactile sensor' --embedding openai
# local BGE optional:
# pip install -e '.[local-dense]'
# python -m techintel.retrieval_enhanced 'robot gripper tactile sensor' --embedding bge-m3 --rerank
```

**24 小时执行清单 / 简历用语 / 面试脚本：** [CANDIDATE_24H_EXECUTION_CN.md](docs/CANDIDATE_24H_EXECUTION_CN.md)。

---

## 1. 五分钟本地运行（macOS、Windows、Linux）

```bash
# 下载本对话提供的 ZIP，解压到某个文件夹后：
cd techintel-agent
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1
pip install -e '.[dev]'
pytest -q
python -m techintel.evaluation --save results/metrics.json
techintel '硫化物电解质为什么对湿度敏感？'
uvicorn techintel.app:app --reload
```

浏览器打开 **http://127.0.0.1:8000/docs** （交互式 API 文档），试用：

```json
POST /ask
{"query":"LLZO 氧化物固态电解质的加工挑战是什么？", "mode":"offline", "session_id":"interview"}
```

响应有 `answer`、`sources`、`evidence`、`trace`、`latency_ms` 和 `abstained`。离线模式是**原文证据摘录**，不是 LLM 生成的答案。

也可以验证记忆（独立 session）：

```json
POST /memory
{"session_id":"interview", "note":"User wants to compare LLZO and sulfides"}
```

随后 `GET /memory/interview` 查询。默认内存型 SQLite，重启进程丢失；想持久化设置 `TECHINTEL_MEMORY_DB=./notes.sqlite`（切勿上传敏感内容）。

## 2. 专利专项增强（24 小时 JD 定向版本）

本次升级新增 `patents.py`、`bm25.py`、`patent_eval.py`，以及 4 个**完全虚构、明确标注不是实际专利**的教学记录。包含基本英文 Claims 依赖解析、权利要求词法覆盖率对比、受控的本地专利检索、独立 BM25 稀疏检索和 RRF 融合函数。新增 REST API：

```json
POST /patents/search
{"query":"robot gripper tactile sensor", "source":"local", "retrieval":"bm25", "k":3}

POST /patents/claims/parse
{"claims_text":"1. A robotic gripper with a tactile sensor.\n2. A gripper of claim 1, with a controller."}

POST /patents/claims/compare
{"target_claims":"1. A device with a sensor.", "reference_claims":"1. A device with a sensor and camera."}
```

额外提供一个 MCP 服务中的 3 个专利工具：`search_patents`、`parse_patent_claims`、`compare_patent_claims`。**此处的 comparison 只衡量词项覆盖，不是法律查新、专利有效性、侵权或 FTO 判断。**

可选 `PATENTSVIEW_API_KEY` 读取美国真实专利元数据，但新申请 Key 可能被服务提供方限制；接口没有在本环境做真实连通验证。详见 [专利冲刺方案及求职真实边界](docs/PATENT_UPGRADE_CN.md)。

单独检查专利示例评测：

```bash
python -m techintel.patent_eval
```

由于只有 **4 篇虚构专利 + 16 道同源作者编写的问题**，结果属于工程冒烟测试，不能外推为真实业务表现。尚未完成 BGE-M3 / BGE-Reranker、真实 dense+sparse 融合效果评测、LoRA 训练或生产上线。

## 3. 技术架构与功能范围

```
Client (CLI / REST / optional MCP host)
      │
FastAPI / Agent Orchestrator
      ├── offline: retrieve → cite → extractive evidence preview
      └── llm: Responses API → model-selected tools → citation ID guard
                    │
             tool whitelist
              ├─ search_knowledge → word+char TF-IDF chunks
              ├─ get_document → local MD/TXT/text-PDF
              ├─ save_memory → session SQLite
              └─ recall_memory

Optional: VectorIndex uses remote text-embedding-3-small + cosine similarity,
          separate experimental path, not part of offline benchmark.
MCP v2: defines search, read, note and patent search/parse/compare tools via stdio; client interoperability still requires real host validation.
```

See [PatentPulse JD plan](docs/PATENT_UPGRADE_CN.md), [design decisions](docs/ARCHITECTURE.md), [Interview guide](docs/INTERVIEW_GUIDE_CN.md), [24-hour SOP](docs/SOP_24H_CN.md), [CV bullets](docs/RESUME_BULLETS_CN.md), [Skill specification](skills/technical-intel/SKILL.md).

**技术术语准确性**：当前混合检索是 TF-IDF word + character n-gram，不是 embedding-based hybrid search，也没有 cross-encoder reranker。`vector.py` 提供真实 embedding API 可选实现，但离线演示不需要 Key。

## 4. 测评：基线对照（仅 synthetic demo）

本仓库默认加载 23 份资料（12 篇 synthetic 技术笔记 + 4 篇真实论文出处摘要 + 4 篇虚构专利教学记录 + 3 篇真实专利出处的人工摘要），有 20 道人工撰写测试题：16 个可回答、4 个知识库外问题。比较 `word`、`hybrid` 与 `bm25` 检索：

| Retriever | Hit@3 / 16 | MRR / 16 | Correct abstention / 4 |
|---|---:|---:|---:|
| word-only TF-IDF | 0.75 | 0.7188 | 1.00 |
| word + character TF-IDF | **1.00** | **0.9688** | **1.00** |
| BM25 （词/中文二字组） | **0.9375** | **0.9375** | **0.25** |

**这不是 LLM 答案正确率**。问题由项目作者按自制文档编写，同源偏差显著。结果主要用于保证工程管道正常运转、观察中文词级检索失效；不能拿去宣称“真实文献检索达到 100%”。完整数值和失败案例在 `results/metrics.json`，可重新执行评价脚本核对。

下一阶段须：替换真实且可合法处理的论文/专利资料，建立独立标注的 50–100 条问题；在相同语料和 splits 上比较 lexical、embedding、rerank；增加基于证据的 claim-level answer evaluation。

## 5. 使用自己的真实资料

1. 把 `.md` / `.txt` / **可提取文字**的 `.pdf` 放入 `corpus/your_docs/`，原始数据不会随 Git 自动上传（默认被 `.gitignore` 忽略）。
2. 保证文件名在 demo/patents_demo/source_notes/your_docs 四个目录内全局唯一，文件名 stem 用作 `doc_id`。可用 OpenAlex/Crossref/论文原文元数据人工登记真实出处（此项目并未自动校验 DOI）。
3. 重启服务重新建立索引，然后执行 `/documents` 检查是否加载成功。
4. 在 `evals/gold.jsonl` 或 `evals/patent_gold.jsonl` 添加提问与 gold doc IDs，严禁把生成答案当作人工金标。
5. 对扫描版 PDF、表格、公式必须人工验证文本提取。此项目不做 OCR。

## 6. 开启真实 LLM 工具调用（可选，按 API 计费）

```bash
# macOS / Linux; Windows 也可使用 PowerShell 的 $env:OPENAI_API_KEY=...
export OPENAI_API_KEY='YOUR_API_KEY'
export OPENAI_MODEL='gpt-5.4-mini'
techintel 'Compare sulfide and oxide electrolytes using tool evidence' --mode llm
```

真实 LLM 模式包含模型驱动 function calling 循环。采用 OpenAI **Responses API**，调用 `search_knowledge`、`get_document`、`save_memory`、`recall_memory` 等 allowlisted 工具，上限 5 轮，并验证最终 `[source:doc_id]` 是否属于检索证据。**验证 ID 是存在性检查，不代表所引用内容在事实上支持该断言**。

公开 API/Model 文档：[Responses function calling](https://developers.openai.com/api/docs/guides/function-calling)、[Embedding](https://developers.openai.com/api/docs/guides/embeddings)。

## 7. 向量检索实验（可选，按 API 计费）

```bash
export OPENAI_API_KEY='YOUR_API_KEY'
python -m techintel.vector 'LLZO ceramic grain boundary resistance'
```

该实验使用 `text-embedding-3-small` 产生真实密集向量，在内存中执行余弦相似度检索。它不是 FAISS、Chroma 或 Milvus 向量数据库，且目前未用真实 API Key 进行网络端到端测试。可以通过 `tests/test_vector.py` 的假 embedding backend 查看向量相似度和排序逻辑；这个 mock 测试不证明语义搜索质量。

## 8. MCP tool server（可选）

```bash
pip install -e '.[mcp]'
python -m techintel.mcp_server
```

使用官方 MCP Python SDK **2.x**（`from mcp.server import MCPServer`）。通过 stdio 向兼容 MCP host 暴露 `search_research`、`read_research_document`、`remember_research_note`。启动后终端看起来停住是正常的——它在等待 MCP 客户端连接或工具请求。可使用 [MCP Inspector](https://github.com/modelcontextprotocol/inspector) 执行 tool smoke test，保留真实截屏。此环境未与外部 MCP 客户端完成连通与工具调用验证。

## 9. Docker (有 Docker 才能实际验证)

```bash
docker build -t techintel-agent:demo .
docker run --rm -p 8000:8000 techintel-agent:demo
# open http://localhost:8000/docs
```

Dockerfile 定义构建流程，**不等于生产上线**。真实上线需补鉴权、会话隔离、审计、访问控制、模型密钥管理、OpenTelemetry、限流和监控。

## 10. 项目目录

```
techintel-agent/
├── src/techintel/agent.py       # fixed offline flow + real LLM tool-call loop
├── src/techintel/retrieval.py   # chunking + bilingual TF-IDF
├── src/techintel/patents.py     # patent search + claims parse/compare
├── src/techintel/bm25.py        # BM25 sparse + RRF fusion helper
├── src/techintel/patent_eval.py # patent synthetic smoke benchmark
├── src/techintel/vector.py      # optional paid embedding retrieval
├── src/techintel/memory.py      # session SQLite memory
├── src/techintel/app.py         # FastAPI + OpenAPI
├── src/techintel/mcp_server.py  # optional MCP v2 server
├── src/techintel/evaluation.py  # Hit@3 / MRR / OOD
├── tests/                      # deterministic automated tests
├── corpus/demo/                # 12 synthetic teaching docs (NOT real papers)
├── corpus/patents_demo/       # 4 entirely fictional patent teaching notes
├── corpus/source_notes/        # 4 paraphrased published-paper source notes
├── corpus/your_docs/           # untracked user-supplied documents
├── evals/gold.jsonl            # 20 author-written retrieval evaluation queries
├── evals/patent_gold.jsonl     # 16 synthetic patent example queries
├── results/metrics.json        # reproducible demo numbers
├── results/patent_metrics.json # synthetic patent smoke benchmark
├── skills/technical-intel/     # task-specific skill instruction
├── docs/                       # architecture / CV / interview / SOP
├── .github/workflows/ci.yml
└── Dockerfile
```

## 11. 项目边界与求职诚信

- This repository is an **educational, independently reproducible MVP**, not a production system.
- If you use its bullets on your CV, **run it yourself and be prepared to explain the code and limitations**.
- Four synthetic patent teaching records were used. No real patents were searched or validated; no legal analysis or novelty conclusion is asserted.
- No online business customers, paid users or deployed SLA.

**相关官方资料**：[MCP Python SDK](https://py.sdk.modelcontextprotocol.io/) · [FastAPI Docker](https://fastapi.tiangolo.com/deployment/docker/) · [Ragas evaluation](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/)
