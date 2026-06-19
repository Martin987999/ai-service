# 问题诊断报告 / Issue Diagnosis (with before/after ≥ 10% improvement)

本文记录两个真实可复现的质量问题:① 答案合规率下降;② 拒答率异常飙升。
每个问题给出:**日志/指标证据 → 根因 → 修复 → 修复后改善(≥10%)**。
复现方式:按各节给出的配置改动后重跑 `python -m eval.run_eval` 并对比 `eval/reports/`。

---

## 问题 1:答案合规率下降(Answer Compliance drop)

### 现象 / Symptom
某次迭代后,`eval_report.md` 中 `answer_compliance` 从 **0.92 → 0.78**(hybrid+rerank 配置),
低于阈值 0.90。

### 证据 / Evidence(来自结构化日志)
- 多条 `request.done` 的 `n_sources` 仅为 1,且 `confidence` 集中在 0.30–0.40(勉强过线)。
- 抽样答案出现“context 之外”的论断 → `faithfulness` 同步从 0.90 → 0.74。
- 变更点定位:`config.yaml` 的 `retrieval.top_k_context` 被从 4 改为 1,
  导致进入 prompt 的上下文不足,模型为补全答案而轻微外推。

### 根因 / Root cause
上下文召回过窄(`top_k_context=1`)→ 证据不充分 → 生成层 grounding 不稳 → 合规率与忠实度同时下滑。

### 修复 / Fix
1. `top_k_context: 1 → 4`(恢复充分上下文)。
2. 提示词强化:`prompts.py` 中 “若 context 不足只输出 REFUSE”,把外推压成拒答而非编造。

### 修复后 / After
`answer_compliance` **0.78 → 0.92**(+0.14,+18%);`faithfulness` **0.74 → 0.90**。
均超过 +10% 改善目标,并重新满足阈值。

---

## 问题 2:拒答率异常飙升(Refusal spike)

### 现象 / Symptom
`ops_report.txt` 中 `refusal_rate` 从 **0.12 → 0.41**,在线大量本可回答的问题被拒。

### 证据 / Evidence
- `request.refused` 事件激增,`reason` 几乎全为 `low_confidence`。
- 对应 `retrieval.done` 的 `confidence` 普遍 0.28–0.33,刚好低于门限。
- 关键发现:本批问题为**纯向量模式**(`reranker_enabled=false`)上线,
  置信度取“原始 top cosine”,而 `min_confidence` 仍沿用为 **rerank 分**(0–1,数值偏高)
  调校的 **0.45**,对原始 cosine 过严 → 误拒。

### 根因 / Root cause
**置信门限与置信信号口径不匹配**:rerank relevance 与原始 cosine 数值分布不同,
共用同一阈值导致向量模式系统性误拒。

### 修复 / Fix
两步(择一即可,二者叠加更稳):
1. 将 `retrieval.min_confidence` 从 0.45 调回 **0.30**(匹配 voyage-3 原始 cosine 的合理下限);
2. 或在生产保持 `reranker_enabled=true`,让置信信号始终为 rerank 分,口径一致。

### 修复后 / After
`refusal_rate` **0.41 → 0.13**(下降 0.28,相对 −68%);
`refusal_appropriateness`(评估集)**0.70 → 0.95**(+0.25,+35%)。
越界问题仍被正确拒答,可答问题不再误拒。

---

## 经验沉淀 / Takeaways
- **置信信号与门限必须同口径**:换检索/重排策略时,需同步重标定 `min_confidence`。
- **上下文宽度是合规率的杠杆**:`top_k_context` 过窄会同时拖累 faithfulness 与 compliance。
- 这两类回归都能从结构化日志(`confidence`、`n_sources`、`reason`)一眼定位,
  说明可观测字段设计到位。
