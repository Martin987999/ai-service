#!/usr/bin/env bash
# One-click evaluation (Linux/macOS/Git-Bash)
# 一键评估:建语料 -> 建索引 -> 跑三配置评估 -> 生成运维报告
# 用法: ./run_eval.sh [builtin|rgb]
set -euo pipefail
SOURCE="${1:-builtin}"
PY="${PYTHON:-python}"

echo "==> [1/4] build corpus + eval set (source=$SOURCE)"
"$PY" -m scripts.make_corpus --source "$SOURCE"

echo "==> [2/4] ingest -> build indices"
"$PY" -m scripts.ingest

echo "==> [3/4] run evaluation (vector / hybrid / hybrid+rerank)"
"$PY" -m eval.run_eval

echo "==> [4/4] ops report from logs"
"$PY" -m eval.report

echo "Done. See eval/reports/ for eval_report.md, eval_results.csv, ops_report.txt"
