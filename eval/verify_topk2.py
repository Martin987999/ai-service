# -*- coding: utf-8 -*-
"""Verify the corpus-size hypothesis: does context_precision improve with a smaller top_k_context?

real_report 显示三种配置 context_precision 均未达 0.70(语料只有 10 块,top_k_context=4 取回偏宽)。
本脚本临时把 top_k_context 改成 2,只重测 context_precision(复用已有真实答案,不重新生成),
验证"精确率低是语料规模/取回宽度问题,不是检索算法缺陷"这一假设。

用法:
  python -m eval.verify_topk2 --rpm 3
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rpm", type=float, default=3)
    ap.add_argument("--top-k-context", type=int, default=2)
    ap.add_argument("--rows", default="eval/reports/eval_rows.csv")
    args = ap.parse_args()

    s = load_settings()
    pipe = build_pipeline(s)
    pipe.cache.semantic_enabled = False
    # override: only affects how many already-ranked candidates get sliced into context —
    # does not change what gets fetched/reranked, just the final cutoff
    pipe.retriever.cfg["top_k_context"] = args.top_k_context
    judge_model = s.get("models", "generation", "judge_model")

    items_by_id = {it["id"]: it for it in load_eval(s.get("paths", "eval_dataset"))}
    rows = load_rows(args.rows)

    print(f"[verify] top_k_context={args.top_k_context} (was 4 in the real run)\n")
    results = []
    for cfg_name, cfg in CONFIGS.items():
        cfg_rows = [r for r in rows if r["config"] == cfg_name
                    and r["refused"] == "False" and r["out_of_scope"] == "False"]
        throttle = voyage_calls_per_item(cfg) * (60.0 / args.rpm) * 1.05 if args.rpm > 0 else 0.0
        ctxp = []
        print(f"[verify] {cfg_name}: {len(cfg_rows)} items, throttle={throttle:.1f}s/item", flush=True)
        for r in cfg_rows:
            it = items_by_id[r["id"]]
            outcome = pipe.retriever.retrieve(it["question"], mode=cfg["mode"], rerank=cfg["rerank"])
            ctx_texts = [c.text for c in outcome.chunks]  # already sliced to top_k_context=2
            score = M.context_precision(pipe.llm, judge_model, it["question"], ctx_texts)
            ctxp.append(score)
            print(f"  - {r['id']}: n_chunks={len(ctx_texts)} context_precision={round(score,3)}",
                  flush=True)
            if throttle:
                time.sleep(throttle)
        avg = round(_avg(ctxp), 4)
        results.append({"config": cfg_name, "context_precision_k2": avg})
        print(f"[verify] {cfg_name} done: context_precision(k=2)={avg}\n", flush=True)

    print("=== Comparison: context_precision at top_k_context=4 (real run) vs k=2 ===")
    baseline = {"vector": 0.5, "hybrid": 0.3438, "hybrid_rerank": 0.5}
    print("config | k=4 (real) | k=2 (this run)")
    for r in results:
        b = baseline.get(r["config"], None)
        print(f"{r['config']:14s} | {b} | {r['context_precision_k2']}")

    out = Path(s.get("paths", "report_dir"))
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "verify_topk2.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["config", "context_precision_k4_real", "context_precision_k2"])
        for r in results:
            w.writerow([r["config"], baseline.get(r["config"]), r["context_precision_k2"]])
    print(f"\n[verify] -> {out/'verify_topk2.csv'}")


if __name__ == "__main__":
    main()
