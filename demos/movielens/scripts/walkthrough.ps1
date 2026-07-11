# MovieLens Ranker Agent 端到端 walkthrough（仓库根或 demos/movielens 下执行）
$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..\..\..")
Set-Location $Root
Write-Host "== leaves MovieLens walkthrough ==" -ForegroundColor Cyan
Write-Host "root: $Root"

function Invoke-Agent([string[]]$Args) {
    Write-Host "`n>> agent $Args" -ForegroundColor Yellow
    & go run ./demos/movielens/cmd/agent @Args
    if ($LASTEXITCODE -ne 0) { throw "agent failed: $Args" }
}

Invoke-Agent @("status")
Invoke-Agent @("prepare")
Invoke-Agent @("train", "-objective", "rank:ndcg", "-rounds", "20", "-depth", "4")
Invoke-Agent @("eval")
Invoke-Agent @("recommend", "-group", "0", "-topk", "10")

Write-Host "`n== done ==" -ForegroundColor Green
Write-Host "metrics: demos/movielens/out/metrics_train.json"
Write-Host "recommend: demos/movielens/out/recommend_g0.json"
Write-Host "tutorial: demos/movielens/TUTORIAL.md"
