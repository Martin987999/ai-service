# Evaluation Report (FULL, real) / 完整真实评估报告

> 真实 Voyage 检索 + 真实 Claude 生成 + 真实 Claude judge(sonnet-4-6),10 题 x 3 配置,全部真实 API 调用,无 mock。

| config | faithfulness | context_precision | answer_compliance | style_consistency | refusal_appropriateness | refusal_rate | p50_latency_ms | p95_latency_ms | input_tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| vector | 1.0 | 0.5 | 1.0 | 0.875 | 1.0 | 0.2 | 2371.91 | 3419.35 | 5925 |
| hybrid | 1.0 | 0.3438 | 1.0 | 0.875 | 1.0 | 0.2 | 2801.25 | 32817.33 | 5969 |
| hybrid_rerank | 1.0 | 0.5 | 1.0 | 1.0 | 1.0 | 0.2 | 2784.0 | 5381.83 | 6085 |

## Thresholds (global constraints)

- Faithfulness ≥ 0.85; Context Precision ≥ 0.70
- Answer Compliance ≥ 90%; Refusal Appropriateness ≥ 90%; Style Consistency ≥ 0.85