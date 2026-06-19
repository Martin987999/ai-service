# One-click evaluation (Windows PowerShell)
# 一键评估:建语料 -> 建索引 -> 跑三配置评估 -> 生成运维报告
# 用法: .\run_eval.ps1   [-Source builtin|rgb]
param(
  [string]$Source = "builtin"
)
$ErrorActionPreference = "Stop"
$py = "python"

Write-Host "==> [1/4] build corpus + eval set (source=$Source)" -ForegroundColor Cyan
& $py -m scripts.make_corpus --source $Source

Write-Host "==> [2/4] ingest -> build indices" -ForegroundColor Cyan
& $py -m scripts.ingest

Write-Host "==> [3/4] run evaluation (vector / hybrid / hybrid+rerank)" -ForegroundColor Cyan
& $py -m eval.run_eval

Write-Host "==> [4/4] ops report from logs" -ForegroundColor Cyan
& $py -m eval.report

Write-Host "Done. See eval/reports/ for eval_report.md, eval_results.csv, ops_report.txt" -ForegroundColor Green
