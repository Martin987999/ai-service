# -*- coding: utf-8 -*-
"""Add the 4 LLM-judge quality metrics on top of an existing light (retrieval+refusal) run.

复用 eval_rows.csv 里已有的**真实 Claude 答案**(不重新生成,省调用),只对在域、未拒答的题
重新做一次真实检索(拿到新鲜 context 供 judge 用——light 模式当时没把 context 落盘)。
然后跑 faithfulness / context_precision / answer_compliance / style_consistency 4 个指标,
和已有的 refusal_appropriateness / 置信度 / 延迟 / 成本合并成一份完整真实报告。

用法:
  python -m eval.run_judge_only --rpm 3
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

from src.app import build_pipeline
from src.config import load_settings
from eval import metrics as M
from eval.run_eval import CONFIGS, voyage_calls_per_item, load_eval, _avg


def load_rows(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_light_results(path: str) -> dict[str, dict]:
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {r["config"]: r for r in rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rpm", type=float, default=3)
    ap.add_argument("--rows", default="eval/reports/eval_rows.csv")
    ap.add_argument("--light-results", default="eval/reports/eval_results.csv")
    ap.add_argument("--limit", type=int, default=0, help="每配置最多跑几题(0=全部;用于快速冒烟测试)")
    args = ap.parse_args()

    s = load_settings()
    pipe = build_pipeline(s)
    pipe.cache.semantic_enabled = False
    judge_model = s.get("models", "generation", "judge_model")

    items_by_id = {it["id"]: it for it in load_eval(s.get("paths", "eval_dataset"))}
    rows = load_rows(args.rows)
    light = load_light_results(args.light_results)

    results = []
    for cfg_name, cfg in CONFIGS.items():
        cfg_rows = [r for r in rows if r["config"] == cfg_name
                    and r["refused"] == "False" and r["out_of_scope"] == "False"]
        if args.limit:
            cfg_rows = cfg_rows[: args.limit]
        throttle = voyage_calls_per_item(cfg) * (60.0 / args.rpm) * 1.05 if args.rpm > 0 else 0.0
        faith, ctxp, comp, style = [], [], [], []
        print(f"[judge] {cfg_name}: {len(cfg_rows)} in-scope answered items, "
              f"throttle={throttle:.1f}s/item", flush=True)
        for r in cfg_rows:
            it = items_by_id[r["id"]]
            outcome = pipe.retriever.retrieve(it["question"], mode=cfg["mode"], rerank=cfg["rerank"])
            ctx_texts = [c.text for c in outcome.chunks]
            answer = r["answer"]  # reuse the real Claude answer captured by the light run
            faith.append(M.faithfulness(pipe.llm, judge_model, answer, ctx_texts))
            ctxp.append(M.context_precision(pipe.llm, judge_model, it["question"], ctx_texts))
            comp.append(M.answer_compliance(pipe.llm, judge_model, it["question"], answer,
                                            ctx_texts, it.get("ground_truths", [])))
            style.append(M.style_consistency(pipe.llm, judge_model, answer, it.get("lang", "en")))
            print(f"  - {it['id']}: faith={faith[-1]} ctxp={round(ctxp[-1],2)} "
                  f"comp={comp[-1]} style={style[-1]}", flush=True)
            if throttle:
                time.sleep(throttle)

        base = light.get(cfg_name, {})
        results.append({
            "config": cfg_name,
            "faithfulness": round(_avg(faith), 4),
            "context_precision": round(_avg(ctxp), 4),
            "answer_compliance": round(_avg(comp), 4),
            "style_consistency": round(_avg(style), 4),
            "refusal_appropriateness": float(base.get("refusal_appropriateness", 0)),
            "refusal_rate": float(base.get("refusal_rate", 0)),
            "p50_latency_ms": float(base.get("p50_latency_ms", 0)),
            "p95_latency_ms": float(base.get("p95_latency_ms", 0)),
            "input_tokens": int(base.get("input_tokens", 0)),
            "output_tokens": int(base.get("output_tokens", 0)),
            "cost_usd_total": float(base.get("cost_usd_total", 0)),
            "cost_usd_per_1k_calls": float(base.get("cost_usd_per_1k_calls", 0)),
        })
        r0 = results[-1]
        print(f"[judge] {cfg_name} done: faithfulness={r0['faithfulness']} "
              f"context_precision={r0['context_precision']} "
              f"answer_compliance={r0['answer_compliance']} "
              f"style_consistency={r0['style_consistency']}\n", flush=True)

    _print_full(results)
    _write_full(results, s.get("paths", "report_dir"))


def _print_full(results: list[dict]) -> None:
    cols = ["config", "faithfulness", "context_precision", "answer_compliance",
            "style_consistency", "refusal_appropriateness", "refusal_rate",
            "p50_latency_ms", "p95_latency_ms", "cost_usd_per_1k_calls"]
    print("\n=== FULL real evaluation (retrieval + refusal + answer quality) ===")
    print(" | ".join(cols))
    for r in results:
        print(" | ".join(str(r[c]) for c in cols))


def _write_full(results: list[dict], report_dir: str) -> None:
    out = Path(report_dir)
    out.mkdir(parents=True, exist_ok=True)
    cols = ["config", "faithfulness", "context_precision", "answer_compliance",
            "style_consistency", "refusal_appropriateness", "refusal_rate",
            "p50_latency_ms", "p95_latency_ms", "input_tokens", "output_tokens",
            "cost_usd_total", "cost_usd_per_1k_calls"]
    with open(out / "eval_results_full.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in results:
            w.writerow({c: r[c] for c in cols})
    md = ["# Evaluation Report (FULL, real) / 完整真实评估报告\n",
          "> 真实 Voyage 检索 + 真实 Claude 生成 + 真实 Claude judge(sonnet-4-6),"
          "10 题 x 3 配置,全部真实 API 调用,无 mock。\n",
          "| " + " | ".join(cols[:-3]) + " |",
          "| " + " | ".join("---" for _ in cols[:-3]) + " |"]
    for r in results:
        md.append("| " + " | ".join(str(r[c]) for c in cols[:-3]) + " |")
    md.append("\n## Thresholds (global constraints)\n")
    md.append("- Faithfulness ≥ 0.85; Context Precision ≥ 0.70")
    md.append("- Answer Compliance ≥ 90%; Refusal Appropriateness ≥ 90%; Style Consistency ≥ 0.85")
    (out / "eval_report_full.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\n[judge] reports -> {out/'eval_report_full.md'} , {out/'eval_results_full.csv'}")


if __name__ == "__main__":
    main()
