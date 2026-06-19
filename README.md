# 双语内部知识库 RAG 问答与生成服务
# Bilingual Internal-KB RAG QA & Generative Service

一个面向内部知识库(员工手册 / 合规指南 / 技术规格 / 架构文档)的**多轮 RAG 问答 + 生成**服务。
语料为**中英双语**并含**扫描 PDF**(OCR)。支持**可配置检索模式 + 重排开关**、**生成质量管控**、
**完整可观测性**、**缓存**与**可复现的评估/诊断**。

A multi-turn RAG QA + generative service over a **bilingual (CN/EN)** internal knowledge base
(incl. scanned PDFs). Configurable retrieval modes, optional reranking, generative quality controls,
full observability, caching, and reproducible evaluation — all driven by config, no code change.

> ⚠️ **运行环境说明 / Runtime note**：本仓库在交付环境中**未被执行**(该机器未安装 Python 运行时,
> 仅有 Microsoft Store 占位 shim)。代码已按下文命令组织为可运行结构,并内置 **mock 回退**,
> 使**无 API key 也能离线跑通 `一键评估`**。请在装有 Python 3.10+ 的环境按「快速开始」执行。

---

## 1. 技术栈与选型依据 / Tech Stack & Rationale

| 层 Layer | 选型 Choice | 依据 Rationale |
|---|---|---|
| 服务 Service | FastAPI + asyncio + ThreadPool | 异步 + 信号量限流,满足「单实例 ≥5 并发」 |
| 生成 Generation | **Claude**(Opus 4.8 / Sonnet 4.6 / Haiku 4.5) | 质量最高的最新模型族;按场景分级控成本 |
| Embedding | **Voyage `voyage-3`**(在线,多语言) | Anthropic 官方推荐搭配;`voyage-3` 多语言适配中英混合。**注:Anthropic 本身不提供 embedding 接口** |
| 重排 Rerank | **Voyage `rerank-2`** | 与 embedding 同源,配置即可启停 |
| 向量库 Vector | FAISS `IndexFlatIP`(余弦) | 轻量、可复现、零外部服务 |
| 词法 Lexical | `rank_bm25`(BM25Okapi) | hybrid 模式补充精确匹配 |
| 融合 Fusion | RRF(Reciprocal Rank Fusion) | 无需调权重,跨模态稳健 |
| OCR | pdfplumber → pytesseract(`chi_sim+eng`) | 扫描件回退;按页文本量自动判定 |
| 公开测试集 Dataset | **RGB(中英双语 RAG benchmark)** + 内置离线样例 | 匹配双语 RAG;内置样例保证一键评估可复现 |

> **为何向量用 Voyage 而非 Claude**:Anthropic 不提供 embedding API,官方文档明确推荐 Voyage AI。
> 因此「Claude 生成 + 在线 embedding」= Claude(生成/评估)+ Voyage(向量 + 重排)。

---

## 2. 架构 / Architecture

```
                ┌──────────── FastAPI (/qa /health /metrics) ───────────┐
   query ─────► │  Semaphore(≥5 并发)  +  ThreadPool                   │
                └───────────────────────────┬───────────────────────────┘
                                             ▼
  ┌── Safety ──┐   ┌── Cache ──┐   ┌──────────── Retrieval ────────────┐   ┌── Generation ──┐
  │ injection  │──►│ exact +   │──►│ vector(FAISS)  ┐                  │──►│ Claude (grounded│
  │ detection  │   │ semantic  │   │ bm25 (BM25)    ┘─RRF─► rerank(opt) │   │  + 语言对齐)    │
  └────────────┘   └───────────┘   └──────────────────────────────────┘   └────────┬────────┘
        │                                  │ confidence < min → refuse              ▼
        ▼                                  ▼                                  PII redaction
   refuse(injection)                 refuse(low-conf / OOS)                        │
                                                                                   ▼
   ───────────────────── 结构化 JSON 日志 (trace_id 贯穿) ────────────────►  response + 引用来源
```

数据流落盘:`data/corpus`(语料)→ `scripts.ingest` →`data/index`(FAISS + BM25 + chunks)。

---

## 3. 快速开始 / Quickstart

```bash
# 0) 依赖(Python 3.10+)
pip install -r requirements.txt
# OCR 需系统组件(可选):tesseract-ocr(含 chi_sim) + poppler

# 1) 配置 key(可留空 → 自动 mock 回退,仅用于跑通)
cp .env.example .env      # 填入 ANTHROPIC_API_KEY / VOYAGE_API_KEY

# 2) 一键评估(建语料 → 建索引 → 三配置评估 → 运维报告)
#    Windows:
./run_eval.ps1
#    Linux/macOS/Git-Bash:
./run_eval.sh
#    或手动分步:
python -m scripts.make_corpus --source builtin   # 或 --source rgb 拉公开集
python -m scripts.ingest
python -m eval.run_eval
python -m eval.report

# 3) 启动服务
uvicorn src.service:app --host 0.0.0.0 --port 8000
curl -X POST localhost:8000/qa -H "Content-Type: application/json" \
     -d '{"query":"入职满五年每年有多少天年假?"}'
```

---

## 4. 可配置检索 + 重排开关(改配置不改代码)/ Config-driven retrieval

全部开关在 `config/config.yaml`:

```yaml
retrieval:
  mode: "hybrid"            # vector | hybrid
  reranker_enabled: true    # 重排开关:改配置即启停,无需改代码
  min_confidence: 0.30      # 低于此 → 触发拒答(置信信号为绝对值:rerank 分或原始 cosine)
```

也可**按请求覆盖**:`POST /qa {"query":"...","mode":"vector","rerank":false}`。

---

## 4.5 多轮对话 / Multi-turn

服务支持**多轮会话**:服务端按 `session_id` 维持历史(`src/session.py`),并对依赖上下文的跟进问题做
**历史感知改写**(`src/rewrite.py`):把"那十年呢?"改写成可独立检索的"入职满十年每年有多少天年假?"
(用 `cheap_model`=haiku 控成本,无 key 时启发式回退)。检索用改写后的独立问题,生成时把近几轮历史
传入 prompt 以消解指代、保持连贯,但**答案仍严格 grounding 于检索上下文**。

```bash
# 1) 开一个会话
SID=$(curl -s -X POST localhost:8000/session | jq -r .session_id)
# 2) 第一轮
curl -s -X POST localhost:8000/qa -H "Content-Type: application/json" \
     -d "{\"query\":\"入职满五年每年有多少天年假?\",\"session_id\":\"$SID\"}"
# 3) 跟进问题(自动改写为独立问题再检索)
curl -s -X POST localhost:8000/qa -H "Content-Type: application/json" \
     -d "{\"query\":\"那十年呢?\",\"session_id\":\"$SID\"}"
# 也支持无状态:直接在请求里带 history:[{role,content},...]
```

离线验证:`python tests/test_multiturn.py`(强制 mock,无需 key)——验证会话记忆 + 改写 + 历史入 prompt。

## 5. 模型选型与成本 / Model Selection & Cost

价表(Claude,每 1M token,2026):Opus 4.8 `$5/$25`、Sonnet 4.6 `$3/$15`、Haiku 4.5 `$1/$5`。

**分级使用 / Tiered usage**(`models.generation`):
- `answer_model = claude-opus-4-8`:主答,质量优先。
- `judge_model = claude-sonnet-4-6`:LLM-as-judge **与主答解耦**,降低自评偏差,且更省。
- `cheap_model = claude-haiku-4-5`:高并发的简单意图分类/改写。

**每 1000 次 QA 成本估算 / Cost per 1,000 calls**(单次约 ~900 输入 + ~80 输出 token):

| 主答模型 | 输入成本 | 输出成本 | 生成/1k | + Voyage 向量+重排/1k | 合计/1k(约) |
|---|---|---|---|---|---|
| Opus 4.8 | 900/1e6×$5=$0.0045 | 80/1e6×$25=$0.0020 | **$6.5** | ~$0.2 | **~$6.7** |
| Sonnet 4.6 | $0.0027 | $0.0012 | **$3.9** | ~$0.2 | **~$4.1** |
| Haiku 4.5 | $0.0009 | $0.0004 | **$1.3** | ~$0.2 | **~$1.5** |

> 取舍 / Trade-offs:**质量** Opus≈Sonnet>Haiku;**成本/延迟** Haiku<Sonnet<Opus。
> 默认 Opus 4.8 保质量;高并发可切 Sonnet 省约 40%;简单问答可下沉 Haiku。**缓存命中** 该次成本≈0。
> 评估期 LLM-judge 成本另计(仅离线,非线上),用 Sonnet 控成本。

---

## 6. 质量指标与评估方法 / Quality Metrics & Methodology

`python -m eval.run_eval` 对 **vector / hybrid / hybrid+rerank** 三配置在评估集上量化:

| 指标 | 评估方法 | 阈值(全局 / 进阶) |
|---|---|---|
| **Faithfulness 忠实度** | LLM-as-judge 判定答案每条论断是否被检索上下文支持(0/1) | ≥ 0.85 |
| **Context Precision 上下文精确率** | 逐块 LLM 相关性判定,取 Top-K 中相关块占比 | ≥ 0.70 |
| **Answer Compliance 答案合规率** | grounding + 命中 ground truth + 未编造(精确命中走快路径,否则 LLM-judge) | ≥ 80% / **≥ 90%** |
| **Style Consistency 风格一致性** | 规则(语言对齐 + 无寒暄)+ LLM 判定简洁/专业 | ≥ 80% / **≥ 0.85** |
| **Refusal Appropriateness 拒答恰当性** | 越界题应拒、可答题不应拒:`refused == out_of_scope` | ≥ 80% / **≥ 90%** |

评估模型与主答模型**解耦**(judge=Sonnet),降低自评偏差。评估集含**越界样本**(预期拒答)。

---

## 7. 检索三配置对比 / Retrieval Comparison

`eval.run_eval` 直接产出对比表(`eval/reports/eval_report.md`)。预期结论:

- **vector-only**:语义召回强,但缩写/专有名词/精确字段(如令牌时效)易漏 → context precision 偏低。
- **hybrid(+BM25, RRF)**:词法补召回,**context precision 提升**,refusal 更稳。
- **hybrid + rerank**:rerank-2 重排后 **faithfulness / compliance 最佳**,置信信号为绝对分,
  **拒答判定最可靠**(详见 `docs/ISSUE_DIAGNOSIS.md` 问题 2)。

> 数值随是否使用真实 Voyage/Claude 而变;mock 回退仅验证链路,不代表真实质量。

---

## 8. 安全 / Security

- **Prompt 注入防御**:输入侧规则检测(覆盖指令/泄露系统提示/越权角色)→ 命中即拒答;
  检索上下文侧 `sanitize_context` 中和间接注入,并在系统提示中声明 `<context>` 内一律为**数据非指令**。
- **PII 处理**:`pii.py` 对**输出**与**日志**脱敏(邮箱/手机/身份证/银行卡/IP)。
- **强制 grounding**:系统提示要求仅依据 `<context>` 作答,不足即输出 `REFUSE`;低置信亦拒答。

---

## 9. 可观测性 / Observability

结构化 **JSON Lines** 日志,`trace_id` 贯穿一次请求各阶段。字段字典见
[`docs/LOG_FIELDS.md`](docs/LOG_FIELDS.md),样例日志见 [`docs/sample_logs.jsonl`](docs/sample_logs.jsonl)。
`GET /metrics` 暴露实时 p50/p95、拒答率、token、成本、缓存命中。
`python -m eval.report` 从日志聚合**运维报告**(p50/p95 延迟、token、缓存命中率、拒答率、合规率近似)。

---

## 10. 缓存 / Caching

`cache.py`:**精确缓存**(对 `规范化query+mode+rerank` 哈希)+ **语义缓存**(query embedding 余弦 ≥ 阈值复用)。
命中路径毫秒级返回,显著降低 p95 与成本;命中率进入运维报告。

---

## 11. 可演进性 / Evolvability

- **检索策略可换**:`mode` / `reranker_enabled` / `top_k_*` / `rrf_k` 全在配置;新增模式只需实现一个分支。
- **模型版本可换**:`models.*` 改字符串即切换(Opus↔Sonnet↔Haiku);成本估算随价表更新。
- **指标/日志可增强**:`eval/metrics.py` 加函数即扩指标;日志加字段即扩监控,字段字典同步维护。
- Provider 层(`providers/`)抽象 + mock 回退,便于在无 key / CI 环境跑通与回归。

---

## 12. 交付物对照 / Deliverables Map

| 要求 | 位置 |
|---|---|
| 完整代码与配置 | `src/`、`config/config.yaml` |
| 一键评估脚本 | `run_eval.ps1` / `run_eval.sh`(= make_corpus → ingest → run_eval → report) |
| 评估报告(含前后对比) | `eval/reports/eval_report.md`、`eval_results.csv`(运行后生成) |
| 日志字段字典 + 样例日志 | `docs/LOG_FIELDS.md`、`docs/sample_logs.jsonl` |
| 运维报告(p50/p95、token、缓存命中、拒答率、合规率) | `python -m eval.report` → `eval/reports/ops_report.txt` |
| 问题诊断(≥2 个,前后改善 ≥10%) | `docs/ISSUE_DIAGNOSIS.md` |
| 检索三配置对比 | `eval/run_eval.py` → 报告表 |

---

## 13. 目录结构 / Layout

```
config/config.yaml          # 全部开关:检索/重排/模型/缓存/安全/路径
src/
  config.py logging_setup.py pii.py safety.py prompts.py cache.py pipeline.py app.py service.py
  session.py(多轮记忆) rewrite.py(历史感知改写)
  providers/ llm.py(Claude) embeddings.py(Voyage)
  ingestion/ loader.py(含 OCR) chunker.py lang.py
  retrieval/ store.py bm25.py retriever.py(RRF + rerank)
scripts/ make_corpus.py ingest.py
eval/ metrics.py run_eval.py report.py datasets/ reports/
tests/ test_multiturn.py(离线多轮验证)
docs/ LOG_FIELDS.md sample_logs.jsonl ISSUE_DIAGNOSIS.md
```
