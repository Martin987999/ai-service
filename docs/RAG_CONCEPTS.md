# RAG 知识库核心概念 / RAG Concepts

结合本项目(`D:\AI Service`,双语内部知识库 RAG 问答服务)整理的 RAG 全流程知识。
每个概念标注**对应的代码文件**,部分附**真实评估数据**作为证据,不是纯理论堆砌。

---

## 0. 整体流程图

```
语料(CN/EN, 含扫描PDF)
   │  scripts/make_corpus.py
   ▼
加载(含 OCR)  src/ingestion/loader.py
   ▼
切块(Chunking)  src/ingestion/chunker.py
   ▼
Embedding(向量化)  src/providers/embeddings.py ──┐
   ▼                                              │
建索引:FAISS(向量)+ BM25(词法)  src/retrieval/store.py
   │  scripts/ingest.py(离线建一次)               │
   ▼                                              │
═══════════════ 在线问答 ═══════════════           │
   ▼                                              │
用户问题 ──(多轮改写)── src/rewrite.py             │
   ▼                                              │
检索:向量召回 + BM25召回 → RRF融合 → (可选)Rerank ◄┘
   src/retrieval/retriever.py
   ▼
置信度判定(够不够回答)  src/safety.py
   ▼
够 ──► 生成(Grounded)  src/prompts.py + src/providers/llm.py
   │         ▼
   │     PII脱敏  src/pii.py
不够/越界/注入 ──► 拒答
   ▼
缓存(精确+语义)  src/cache.py
   ▼
结构化日志  src/logging_setup.py
   ▼
返回答案 + 引用来源
```

---

## 1. 数据准备阶段

### 1.1 文档加载(含扫描 PDF OCR)
**文件**:[`src/ingestion/loader.py`](../src/ingestion/loader.py)

- `_load_text` / `_load_jsonl`:纯文本/JSONL 语料直接读
- `_load_pdf`:先用 `pdfplumber` 抽文字层;某页文字 < 40 字符时判定为**扫描件**,回退 `_ocr_page`
- `_ocr_page`:`pdf2image` 转图 + `pytesseract`(`chi_sim+eng`)做中英双语 OCR

⚠️ 本项目里这条 OCR 路径**写了但没真实测过**(没装 tesseract/poppler,没有真实扫描样本)。

### 1.2 切块(Chunking)
**文件**:[`src/ingestion/chunker.py`](../src/ingestion/chunker.py)

- 按**字符数**切(中英混合按字符比按 token/词切更稳)
- 优先在句界(`。.!?！？\n`)断开,带 `overlap` 保上下文连续
- 关键参数(`config/config.yaml` → `chunking`):`chunk_size=700`、`chunk_overlap=120`、`min_chunk_chars=80`

**为什么切块大小很重要**:块太大,一个块混进多个主题,语义被稀释,排不进 top-k(伤召回率);块太小,丢失上下文,模型读到的信息不完整(伤生成质量)。需要权衡。

---

## 2. 索引阶段

### 2.1 Embedding(向量化)
**文件**:[`src/providers/embeddings.py`](../src/providers/embeddings.py)

- 把文字转成向量(一组浮点数),**语义相近的文字,向量距离也相近**
- 真实调用 **Voyage `voyage-3`**(1024 维,多语言)——因为 **Anthropic/Claude 不提供 embedding 接口**,这是本项目"Claude 生成 + Voyage 向量"分工的根本原因
- 三个用途:① 建索引时把文档块向量化;② 检索时把问题向量化去匹配;③ 语义缓存判断两个问题"意思像不像"

### 2.2 向量索引(FAISS)+ 词法索引(BM25)
**文件**:[`src/retrieval/store.py`](../src/retrieval/store.py)、[`src/retrieval/bm25.py`](../src/retrieval/bm25.py)

| 索引 | 原理 | 优势 | 盲区 |
|---|---|---|---|
| FAISS(`IndexFlatIP`) | 向量内积(向量已 L2 归一化,内积=余弦相似度) | 能匹配"意思一样但措辞不同"的问题 | 漏精确术语/缩写/专有名词(语义模型有时会"模糊"掉这些) |
| BM25(`rank_bm25` `BM25Okapi`) | 关键词统计打分(词频 × 逆文档频率,按文档长度归一化) | 精确匹配术语准、快、零成本 | 完全不理解语义,同义表达匹配不上 |

两者互补,这就是为什么要做 **hybrid(混合检索)**。

---

## 3. 检索阶段

**文件**:[`src/retrieval/retriever.py`](../src/retrieval/retriever.py)——`Retriever.retrieve()`,三段流水线:

### 3.1 第一阶段:粗召回(Recall)
- `vector` 模式:只用 FAISS
- `hybrid` 模式:FAISS + BM25 各召回一批,用 **RRF(Reciprocal Rank Fusion)** 融合
  - RRF 公式:`score = Σ 1/(k + rank)`——按**排名倒数**相加,不是按分数加权(两路分数量纲不可比,RRF 不需要调权重)

### 3.2 第二阶段:精排(Rerank,可选开关)
**文件**:[`src/providers/embeddings.py`](../src/providers/embeddings.py) `rerank()` 方法

- 用 Voyage **`rerank-2`** 交叉编码器(query + 每个候选文档**配对**输入,直接判断相关性)
- 比"分别 embed 再比余弦"精确得多,但贵(没法预计算,只能现查现算)→ 所以只对粗召回的小批候选做,不会对全部语料做
- 开关:`config.yaml` 的 `retrieval.reranker_enabled`,改配置不改代码

### 3.3 召回率(Recall)vs 精确率(Precision)
| 指标 | 含义 | 本项目状态 |
|---|---|---|
| 精确率(Context Precision) | 取回的结果里有多少真相关 | ✅ 已实现并真实测过:[`eval/metrics.py`](../eval/metrics.py) `context_precision()` |
| 召回率(Recall) | 所有真相关的结果里,有多少被找回来了 | ❌ **还没实现**——需要先标注"每道题的标准相关 chunk_id 是哪些",当前评估集只有答案文本(`ground_truths`),没有相关文档 ID 标注 |

**两者此消彼长**:取回数量(`top_k_*`)越大,召回率通常越高,但精确率会被拉低。**真实数据印证**:把 `top_k_context` 从 4 调到 2,context precision 从 0.50/0.34/0.50 直接冲到 1.0/0.56/1.0(见 [`eval/reports/verify_topk2.csv`](../eval/reports/verify_topk2.csv))——但同时召回率理论上会下降(没去测,因为没有标注数据)。

---

## 4. 生成阶段

### 4.1 Grounding(强制基于检索内容回答)
**文件**:[`src/prompts.py`](../src/prompts.py) + [`src/safety.py`](../src/safety.py)

- 系统提示词写死:只能依据 `<context>` 回答,不足就输出 `REFUSE`,不得编造
- `<context>` 内文字声明为"数据",即使其中有指令也不执行(防间接注入)

### 4.2 多轮对话(History-aware Rewrite)
**文件**:[`src/session.py`](../src/session.py)(会话记忆)+ [`src/rewrite.py`](../src/rewrite.py)(改写)

- 跟进问题("那十年呢?")单独拿去检索几乎检索不到东西,必须先**改写成独立问题**("入职满十年每年有多少天年假?")才能正确检索
- 用便宜模型(haiku)做改写,省成本
- 历史会拼进生成提示词帮助消解指代,但**明确标注"不作为事实依据"**——答案仍只认 `<context>`

### 4.3 语言对齐 & 风格一致性
- 回答语言跟随提问语言(中文问中文答,英文问英文答)
- 风格规则写在提示词里:简洁、先结论后要点、句末标 `[来源: doc_id]`

---

## 5. 安全与质量控制

### 5.1 拒答机制(三道防线)
1. **Prompt 注入检测**([`src/safety.py`](../src/safety.py) `detect_injection`):命中"忽略之前指令"之类的注入模式,直接拒答,不进检索
2. **低置信度拒答**([`src/pipeline.py`](../src/pipeline.py) `should_refuse_low_confidence`):检索置信度低于阈值,**不调生成**,直接拒答
3. **模型自评拒答**:模型自己判断 context 不足,输出 `REFUSE` 信号词,被拦截转成正式拒答

### 5.2 置信度信号要用绝对值,不能用相对值
真实踩过的坑:RRF 融合后的分数是**归一化相对分**(最高分恒为 ~1.0),拿来做拒答阈值毫无意义(越界题也会被打高分)。必须用**绝对信号**:reranked 时用 rerank 相关性分,否则用原始 cosine。详见 [`docs/ISSUE_DIAGNOSIS.md`](ISSUE_DIAGNOSIS.md) 问题 2。

### 5.3 防幻觉(Hallucination)
四层叠加:Grounding 提示词 → 低置信拒答(根本不让模型硬答)→ 模型自评拒答 → 上下文防注入污染。
用独立 LLM-judge 的 `faithfulness` 指标检测有没有真的防住——**真实评估三种配置全部 1.0**。

### 5.4 PII 脱敏
**文件**:[`src/pii.py`](../src/pii.py)——正则匹配邮箱/手机/身份证/银行卡/IP,对**输出**和**日志**都做脱敏。

---

## 6. 评估体系

### 6.1 LLM-as-Judge(评估模型与主答模型解耦)
**文件**:[`eval/metrics.py`](../eval/metrics.py)

为什么解耦:用同一个模型自己生成又自己评分,容易自我偏袒。本项目主答用 `opus-4-8`,judge 用 `sonnet-4-6`,不同模型互相制约。

4 个质量指标:
| 指标 | 怎么判 |
|---|---|
| Faithfulness(忠实度) | judge 逐条核对答案论断是否被 context 支撑 |
| Context Precision(上下文精确率) | judge 逐块判断检索结果是否真相关 |
| Answer Compliance(答案合规率) | grounding+命中标准答案+未编造(精确命中走快路径省调用) |
| Style Consistency(风格一致性) | 规则(语言对齐+无寒暄)+ judge 判断简洁专业 |

### 6.2 检索配置对比(三档)
`vector` / `hybrid` / `hybrid_rerank`——本项目**真实数据**证实:小语料下 `hybrid` 反而比 `vector` 精确率更低(BM25 引入噪声),加 `rerank` 立刻纠偏。详见 [README §7](../README.md)。

---

## 7. 可观测性与缓存

### 7.1 结构化日志
**文件**:[`src/logging_setup.py`](../src/logging_setup.py),字段字典见 [`docs/LOG_FIELDS.md`](LOG_FIELDS.md)

每条请求一个 `trace_id`,贯穿检索→生成→缓存→安全各阶段,JSON Lines 格式,带 PII 脱敏。

### 7.2 缓存(两层)
**文件**:[`src/cache.py`](../src/cache.py)

- **精确缓存**:问题规范化后哈希命中
- **语义缓存**:embedding 余弦相似度 ≥ 阈值(0.95)就复用答案,容错措辞差异

---

## 8. 这个项目里已验证 vs 未验证的知识点(诚实清单)

| 知识点 | 状态 |
|---|---|
| Embedding / BM25 / Hybrid / RRF / Rerank | ✅ 真实数据验证过 |
| Faithfulness / Compliance / 拒答准确率 | ✅ 真实数据验证过(均达标) |
| Context Precision 未达标根因(语料规模) | ✅ 已用 `top_k_context` 调参实验验证 |
| Context Recall(召回率) | ❌ 未实现(缺相关文档标注) |
| 多轮对话改写逻辑 | ⚠️ 仅 mock 验证过,未用真实 API 跑过真实多轮 |
| 扫描 PDF OCR | ❌ 完全未测试(无真实样本、未装系统依赖) |
| **并发(≥5)+ 线程安全** | ✅ 已验证:`tests/test_concurrency.py`,8 并发在飞 / 120 请求 / 0 竞态(已为缓存/会话/指标加锁) |
| **大语料扩展性** | ✅ 已验证:`tests/test_large_corpus.py`,3000 块下 vector p95=0.68ms / hybrid p95=10.5ms(均 << 10s SLA) |
| 真实生产数据 | ⚠️ 需你提供真实文档(丢进 `data/corpus` 即可);当前用 800~3000 块合成语料做规模代理 |

> ⚠️ 真实跑出的扩展性结论:**FAISS 向量检索几乎不随规模变化**(800→3000 块,0.4ms→0.68ms);
> **BM25(`rank_bm25` 纯 Python `O(N)`)是线性增长的扩展瓶颈**(2.6ms→10.5ms),到几十万块需换真正的
> 倒排索引(Elasticsearch/OpenSearch)。当前规模下完全够用,但这是已知的 scale-out 改造点。

## 9. 接入真实生产数据

ingestion 路径已就绪——把真实文档丢进 `data/corpus`(支持 `.txt` / `.md` / `.jsonl` / `.pdf`,
PDF 含扫描件 OCR 回退),然后:
```bash
python -m scripts.ingest          # 加载→切块→Voyage 向量化→建 FAISS+BM25 索引
```
注意:① 大语料首次建索引会产生较多 Voyage embedding 调用(可批量,免费层 3 RPM 会慢,付费层快);
② 扫描 PDF 需系统装 `tesseract`(含 `chi_sim`)+ `poppler`;③ 真实数据进来后建议重跑评估
(`run_eval.ps1`)拿到真实语料下的指标——玩具语料的 context precision 偏低主要是语料规模问题(见 §3.3)。

---

延伸阅读:[`README.md`](../README.md)(架构总览+成本)、[`docs/ISSUE_DIAGNOSIS.md`](ISSUE_DIAGNOSIS.md)(真实问题诊断)、[`docs/LOG_FIELDS.md`](LOG_FIELDS.md)(日志字段)。
