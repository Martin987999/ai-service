"""Minimal operations report from structured logs.

从 logs/service.jsonl 聚合生成运维报告(text + CSV):
  p50/p95 延迟、token 用量、缓存命中率、拒答率、答案合规率(近似)。
合规率说明:严格合规率来自 `eval.run_eval`(需 ground truth);本报告给出
运行期近似 = 已答非拒答请求占比,作为线上监控代理指标。

用法:python -m eval.report  [--logs logs/service.jsonl] [--out eval/reports]
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_events(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


def pct(vals, p):
    if not vals:
        return 0.0
    s = sorted(vals)
    k = max(0, min(len(s) - 1, int(round((p / 100) * (len(s) - 1)))))
    return round(s[k], 2)


def build_report(events: list[dict]) -> dict:
    done = [e for e in events if e.get("event") == "request.done"]
    refused = [e for e in events if e.get("event") == "request.refused"]
    cache_hits = [e for e in events if e.get("event") == "cache.hit"]
    starts = [e for e in events if e.get("event") == "request.start"]

    lat = [e["latency_ms"] for e in done if "latency_ms" in e]
    lat += [e["latency_ms"] for e in refused if "latency_ms" in e]
    in_tok = sum(e.get("input_tokens", 0) for e in done)
    out_tok = sum(e.get("output_tokens", 0) for e in done)
    cost = sum(e.get("cost_usd", 0.0) for e in done)

    total = len(starts) or (len(done) + len(refused))
    total = total or 1
    answered = len(done)
    n_refused = len(refused)

    return {
        "total_requests": total,
        "answered": answered,
        "refused": n_refused,
        "refusal_rate": round(n_refused / total, 4),
        "answer_compliance_rate_approx": round(answered / total, 4),
        "cache_hits": len(cache_hits),
        "cache_hit_rate": round(len(cache_hits) / total, 4),
        "p50_latency_ms": pct(lat, 50),
        "p95_latency_ms": pct(lat, 95),
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "cost_usd_total": round(cost, 6),
        "cost_usd_per_1k_calls": round(cost / total * 1000, 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="logs/service.jsonl")
    ap.add_argument("--out", default="eval/reports")
    args = ap.parse_args()

    events = load_events(args.logs)
    rep = build_report(events)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "ops_report.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k, v in rep.items():
            w.writerow([k, v])

    lines = ["Operations Report / 运维报告", "=" * 32]
    for k, v in rep.items():
        lines.append(f"{k:32s}: {v}")
    text = "\n".join(lines)
    (out / "ops_report.txt").write_text(text, encoding="utf-8")
    print(text)
    print(f"\n[report] -> {out/'ops_report.txt'} , {out/'ops_report.csv'}")


if __name__ == "__main__":
    main()
