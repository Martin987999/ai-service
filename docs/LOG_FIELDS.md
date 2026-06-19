# 日志字段字典 / Log Field Dictionary

结构化日志为 **JSON Lines**(每行一个事件),默认输出到 stdout 与 `logs/service.jsonl`。
所有 payload 在落盘/输出前经过 **PII 脱敏**(`safety.pii_redaction_logs=true` 时)。
每条 QA 请求共享一个 `trace_id`,可据此串联检索/生成/缓存/安全各阶段。

Structured logs are **JSON Lines** (one event per line), written to stdout and `logs/service.jsonl`.
Every payload is **PII-redacted** before emission. All events of one QA request share a `trace_id`.

> `docs/sample_logs.jsonl` 取自真实评估运行(真实 Voyage 检索 + 真实 Claude 生成,`is_mock: false`),
> 含一次正常作答、一次低置信拒答、一次注入拒答;唯一例外是 `cache.hit` 示例——评估期关闭了语义缓存
> 以节省 Voyage 配额,没有产生真实命中,该行为手工构造的格式示例,**如实标注**。

## 公共字段 / Common fields (every line)

| 字段 field | 类型 type | 说明 description |
|---|---|---|
| `ts` | string | 本地时间戳 ISO8601(毫秒)Timestamp |
| `level` | string | `INFO` / `WARNING` / `ERROR` |
| `logger` | string | 日志器名,如 `rag.pipeline` |
| `trace_id` | string | 请求级追踪 ID(16 hex)Per-request trace id |
| `event` | string | 事件名(见下)Event name |

## 事件与专有字段 / Events and their fields

### `request.start`
| 字段 | 说明 |
|---|---|
| `query_lang` | 提问语言 `zh`/`en`/`unknown` |
| `mode` | 本次检索模式 `vector`/`hybrid` |
| `rerank` | 是否启用重排 bool |
| `query_preview` | 脱敏后的问题前 120 字符 |
| `multi_turn` | 本轮是否带历史(多轮)bool |

### `query.rewritten`(历史感知改写,仅多轮且发生改写时)
| 字段 | 说明 |
|---|---|
| `original` | 原始跟进问题(脱敏,前 80 字符) |
| `rewritten` | 改写后的独立问题(脱敏,前 120 字符) |

### `retrieval.done`
| 字段 | 说明 |
|---|---|
| `mode` | 实际使用的检索模式 |
| `reranked` | 是否经过重排 |
| `n_chunks` | 进入 prompt 的上下文条数 |
| `confidence` | 绝对置信度(rerank 分 或 原始 top cosine) |
| `n_vector` | 向量初召回数量 |
| `n_candidates` | 融合/重排后候选数 |
| `raw_vec_top` | 向量最高余弦(调试用) |

### `cache.hit`
| 字段 | 说明 |
|---|---|
| `hit_type` | `exact` 精确命中 / `semantic` 语义命中 |
| `latency_ms` | 命中路径端到端耗时 |

### `request.done`(成功作答)
| 字段 | 说明 |
|---|---|
| `refused` | `false` |
| `latency_ms` | 端到端耗时(毫秒) |
| `input_tokens` | 生成调用输入 token |
| `output_tokens` | 生成调用输出 token |
| `cost_usd` | 本次生成成本(USD,按模型价表) |
| `pii_redactions` | 输出中各类 PII 脱敏计数,如 `{"[EMAIL]":1}` |
| `n_sources` | 引用来源数量 |
| `is_mock` | 是否走了 mock provider |

### `request.refused`(拒答)
| 字段 | 说明 |
|---|---|
| `reason` | `prompt_injection` / `low_confidence` / `model_refusal` |
| `confidence` | 触发拒答时的置信度 |
| `mode` | 检索模式 |
| `latency_ms` | 端到端耗时 |

## 用于监控/诊断的派生指标 / Derived monitoring metrics

| 指标 | 来源事件 | 计算 |
|---|---|---|
| p50/p95 延迟 | `request.done` + `request.refused`.`latency_ms` | 分位数 |
| 拒答率 refusal_rate | `request.refused` / `request.start` | 比值 |
| 缓存命中率 cache_hit_rate | `cache.hit` / `request.start` | 比值 |
| token / 成本 | `request.done`.`input_tokens`/`output_tokens`/`cost_usd` | 求和 |
| 合规率(近似) | `request.done` / `request.start` | 已答占比(严格值见评估报告) |

> 运维报告由 `python -m eval.report` 从本日志聚合生成。
