#!/usr/bin/env bash
# One-click evaluation (Linux/macOS/Git-Bash)
# 一键评估:建语料 -> 建索引 -> 跑三配置评估(检索+拒答+4个质量指标) -> 生成运维报告
# 用法: ./run_eval.sh [builtin|rgb] [rpm]
#   rpm: 每分钟最多多少次 Voyage 调用,防止免费层(3 RPM)限速报错(默认 3);
#        已加值卡的付费层可传 0 关闭限速
set -euo pipefail
SOURCE="${1:-builtin}"
RPM="${2:-3}"
PY="${PYTHON:-python}"

echo "==> [1/4] build corpus + eval set (source=$SOURCE)"
"$PY" -m scripts.make_corpus --source "$SOURCE"

echo "==> [2/4] ingest -> build indices"
"$PY" -m scripts.ingest

echo "==> [3/4] run evaluation (vector / hybrid / hybrid+rerank, throttle=${RPM} rpm)"
"$PY" -m eval.run_eval --rpm "$RPM" --no-semantic-cache

echo "==> [4/4] ops report from logs"
"$PY" -m eval.report

echo "Done. See eval/reports/ for eval_report.md, eval_results.csv, eval_rows.csv, ops_report.txt"
