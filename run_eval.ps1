# One-click evaluation (Windows PowerShell)
# 一键评估:建语料 -> 建索引 -> 跑三配置评估(检索+拒答+4个质量指标) -> 生成运维报告
# 用法: .\run_eval.ps1   [-Source builtin|rgb] [-RPM 3]
#   -RPM: 每分钟最多多少次 Voyage 调用,防止免费层(3 RPM)限速报错;已加值卡的付费层可设 0 关闭限速
param(
  [string]$Source = "builtin",
  [double]$RPM = 3
)
$ErrorActionPreference = "Stop"
$py = "python"

Write-Host "==> [1/4] build corpus + eval set (source=$Source)" -ForegroundColor Cyan
& $py -m scripts.make_corpus --source $Source

Write-Host "==> [2/4] ingest -> build indices" -ForegroundColor Cyan
& $py -m scripts.ingest

Write-Host "==> [3/4] run evaluation (vector / hybrid / hybrid+rerank, throttle=$RPM rpm)" -ForegroundColor Cyan
& $py -m eval.run_eval --rpm $RPM --no-semantic-cache

Write-Host "==> [4/4] ops report from logs" -ForegroundColor Cyan
& $py -m eval.report

Write-Host "Done. See eval/reports/ for eval_report.md, eval_results.csv, eval_rows.csv, ops_report.txt" -ForegroundColor Green
