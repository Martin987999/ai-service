"""One-click evaluation.

对比三种检索配置:vector-only / hybrid / hybrid+rerank。
对每个配置在整个评估集上计算:
  faithfulness、context_precision、answer_compliance、style_consistency、
  refusal_appropriateness,以及 p50/p95 延迟、token、成本、缓存命中率、拒答率。
输出:控制台表格 + eval/reports/eval_report.md + eval/reports/eval_results.csv。

用法:
  python -m eval.run_eval                 # 跑全部三配置
  python -m eval.run_eval --configs hybrid_rerank
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

from src.app import build_pipeline
from src.config import load_settings
from eval import metrics as M

CONFIGS = {
    "vector": {"mode": "vector", "rerank": False},
    "hybrid": {"mode": "hybrid", "rerank": False},
    "hybrid_rerank": {"mode": "hybrid", "rerank": True},
}


def load_eval(path: str) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def pct(vals: list[float], p: int) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    k = max(0, min(len(s) - 1, int(round((p / 100) * (len(s) - 1)))))
    return round(s[k], 2)


def voyage_calls_per_item(cfg: dict) -> int:
    """1 query embed (vector/hybrid) + 1 rerank call if reranking — used to size throttling."""
    return 1 + (1 if cfg["rerank"] else 0)


def run_config(pipe, judge_model, items, cfg_name, cfg, throttle: float = 0.0,
              light: bool = False) -> dict:
    """light=True: skip the 4 LLM-judge quality metrics (faithfulness/context_precision/
    answer_compliance/style_consistency) and only validate retrieval + refusal — fewer Claude
    calls, and zero extra Voyage calls beyond what answer() itself uses."""
    faith, ctxp, comp, style, refus = [], [], [], [], []
    conf_in, conf_out = [], []
    lat, in_tok, out_tok, cost = [], 0, 0, 0.0
    refusals = 0
    rows = []

    for it in items:
        q = it["question"]
        # single retrieval: answer() returns context text (avoids a 2nd embed/rerank → Voyage RPM)
        resp = pipe.answer(q, mode=cfg["mode"], rerank=cfg["rerank"], include_context_text=True)
        ctx_texts = [s.get("text", "") for s in resp.sources]
        lat.append(resp.latency_ms)
        u = resp.usage or {}
        in_tok += u.get("input_tokens", 0); out_tok += u.get("output_tokens", 0)
        cost += u.get("cost_usd", 0.0)
        if resp.refused:
            refusals += 1

        oos = bool(it.get("out_of_scope", False))
        refus.append(M.refusal_appropriate(resp.refused, oos))
        (conf_out if oos else conf_in).append(resp.confidence)

        # quality metrics only meaningful for answered, in-scope items
        if not light and not resp.refused and not oos:
            faith.append(M.faithfulness(pipe.llm, judge_model, resp.answer, ctx_texts))
            ctxp.append(M.context_precision(pipe.llm, judge_model, q, ctx_texts))
            comp.append(M.answer_compliance(pipe.llm, judge_model, q, resp.answer, ctx_texts,
                                            it.get("ground_truths", [])))
            style.append(M.style_consistency(pipe.llm, judge_model, resp.answer, it.get("lang", "en")))

        rows.append({"id": it["id"], "config": cfg_name, "refused": resp.refused,
                     "out_of_scope": oos, "confidence": round(resp.confidence, 4),
                     "latency_ms": resp.latency_ms, "answer": resp.answer})

        if throttle:
            time.sleep(throttle)

    n = max(1, len(items))
    return {
        "config": cfg_name,
        "faithfulness": _avg_or_none(faith, light),
        "context_precision": _avg_or_none(ctxp, light),
        "answer_compliance": _avg_or_none(comp, light),
        "style_consistency": _avg_or_none(style, light),
        "refusal_appropriateness": round(_avg(refus), 4),
        "refusal_rate": round(refusals / n, 4),
        "avg_confidence_in_scope": round(_avg(conf_in), 4) if conf_in else None,
        "avg_confidence_out_of_scope": round(_avg(conf_out), 4) if conf_out else None,
        "p50_latency_ms": pct(lat, 50),
        "p95_latency_ms": pct(lat, 95),
        "input_tokens": in_tok, "output_tokens": out_tok,
        "cost_usd_total": round(cost, 6),
        "cost_usd_per_1k_calls": round(cost / n * 1000, 4),
        "cache": pipe.cache.stats(),
        "_rows": rows,
    }


def _avg(xs: list[float]) -> float:
    return statistics.fmean(xs) if xs else 0.0


def _avg_or_none(xs: list[float], light: bool):
    if light:
        return None
    return round(_avg(xs), 4)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="*", default=list(CONFIGS.keys()),
                    choices=list(CONFIGS.keys()))
    ap.add_argument("--light", action="store_true",
                   help="只验证检索+拒答(跳过 4 个 LLM-judge 质量指标,省 Voyage/Claude 调用)")
    ap.add_argument("--rpm", type=float, default=0,
                   help="限速:每分钟最多多少次 Voyage 调用(如免费层 3)。0=不限速")
    ap.add_argument("--no-semantic-cache", action="store_true",
                   help="评估期关闭语义缓存(省一次 query embed/条,配合限速使用)")
    args = ap.parse_args()

    s = load_settings()
    pipe = build_pipeline(s)
    if args.no_semantic_cache:
        pipe.cache.semantic_enabled = False
    # mock embeddings have no real rate limit — throttling them only wastes time.
    # Only respect --rpm when actually hitting the real Voyage API.
    effective_rpm = args.rpm if not pipe.embedder.is_mock else 0
    judge_model = s.get("models", "generation", "judge_model")
    items = load_eval(s.get("paths", "eval_dataset"))
    print(f"[eval] {len(items)} items | judge={judge_model} | light={args.light} | "
          f"llm_mock={pipe.llm.is_mock} embed_mock={pipe.embedder.is_mock} | "
          f"rpm={effective_rpm}{' (forced 0: mock embedder)' if pipe.embedder.is_mock and args.rpm else ''}\n")

    results = []
    for name in args.configs:
        cfg = CONFIGS[name]
        throttle = 0.0
        if effective_rpm > 0:
            calls = voyage_calls_per_item(cfg)
            throttle = calls * (60.0 / effective_rpm) * 1.05  # 5% safety margin
        # fresh cache per config so cache hits don't skew cross-config latency
        pipe.cache.hits = pipe.cache.misses = 0
        pipe.cache._store.clear()
        t0 = time.perf_counter()
        print(f"[eval] {name:14s} starting (throttle={throttle:.1f}s/item) ...", flush=True)
        r = run_config(pipe, judge_model, items, name, cfg, throttle=throttle, light=args.light)
        print(f"[eval] {name:14s} done in {time.perf_counter()-t0:.1f}s")
        results.append(r)

    _print_table(results, args.light)
    _write_reports(results, s.get("paths", "report_dir"), args.light)


_QUALITY_COLS = ["faithfulness", "context_precision", "answer_compliance", "style_consistency"]
_BASE_COLS = ["config", "refusal_appropriateness", "refusal_rate",
              "avg_confidence_in_scope", "avg_confidence_out_of_scope",
              "p50_latency_ms", "p95_latency_ms", "cost_usd_per_1k_calls"]


def _cols(light: bool) -> list[str]:
    return _BASE_COLS if light else ["config", *_QUALITY_COLS, *_BASE_COLS[1:]]


def _fmt(v) -> str:
    return "N/A" if v is None else str(v)


def _print_table(results: list[dict], light: bool) -> None:
    cols = _cols(light)
    print("\n=== Retrieval configuration comparison" + (" (light: retrieval+refusal only)" if light else "") + " ===")
    print(" | ".join(c[:22] for c in cols))
    for r in results:
        print(" | ".join(_fmt(r[c]) for c in cols))


def _write_reports(results: list[dict], report_dir: str, light: bool) -> None:
    out = Path(report_dir)
    out.mkdir(parents=True, exist_ok=True)
    cols = _cols(light) + ["input_tokens", "output_tokens", "cost_usd_total"]
    # CSV
    import csv
    with open(out / "eval_results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in results:
            w.writerow([_fmt(r[c]) for c in cols])
    # per-item rows (real answers/confidence/refusal — useful for spot-checking)
    with open(out / "eval_rows.csv", "w", newline="", encoding="utf-8") as f:
        row_cols = ["id", "config", "refused", "out_of_scope", "confidence", "latency_ms", "answer"]
        w = csv.DictWriter(f, fieldnames=row_cols)
        w.writeheader()
        for r in results:
            for row in r.get("_rows", []):
                w.writerow({k: row.get(k) for k in row_cols})
    # Markdown
    md = ["# Evaluation Report / 评估报告\n"]
    if light:
        md.append("> **light 模式**:本次仅验证检索 + 拒答(真实 Voyage + 真实 Claude 生成),"
                  "未跑 4 个 LLM-judge 质量指标(faithfulness/context_precision/answer_compliance/"
                  "style_consistency)以节省 Voyage 免费层调用配额。\n")
    md.append("| " + " | ".join(cols[:-3]) + " |")
    md.append("| " + " | ".join("---" for _ in cols[:-3]) + " |")
    for r in results:
        md.append("| " + " | ".join(_fmt(r[c]) for c in cols[:-3]) + " |")
    md.append("\n## Thresholds (global constraints)\n")
    md.append("- Faithfulness ≥ 0.85; Context Precision ≥ 0.70")
    md.append("- Answer Compliance ≥ 90%; Refusal Appropriateness ≥ 90%; Style Consistency ≥ 0.85")
    (out / "eval_report.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\n[eval] reports -> {out/'eval_report.md'} , {out/'eval_results.csv'} , {out/'eval_rows.csv'}")


if __name__ == "__main__":
    main()
