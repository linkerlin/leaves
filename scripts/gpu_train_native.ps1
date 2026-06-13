# Windows native GPU workflow (no WSL)
# Mode xgb-gpu  : XGBoost CUDA train -> JSON -> leaves inference
# Mode leaves-cpu: leaves multi-thread CPU hist
# Mode check    : verify CUDA / XGBoost / GoMLX xla:cpu

param(
    [ValidateSet("xgb-gpu", "leaves-cpu", "check")]
    [string]$Mode = "check"
)

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

function Test-Cuda {
    Write-Host "=== CUDA driver ===" -ForegroundColor Cyan
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
}

function Test-XGBoostGPU {
    Write-Host "=== XGBoost GPU (Windows native) ===" -ForegroundColor Cyan
    python -c "import xgboost as xgb; d=xgb.DMatrix([[0,1],[2,3]], label=[0,1]); b=xgb.train({'device':'cuda:0','tree_method':'hist','max_depth':2}, d, num_boost_round=2); print('xgb gpu ok, trees=', len(b.get_dump()))"
}

function Test-GoMLX-CPU {
    Write-Host "=== GoMLX xla:cpu (Windows: CPU PJRT only) ===" -ForegroundColor Cyan
    . "$PSScriptRoot\setup_xla_env.ps1"
    if (-not (Test-Path "bin\verify_pjrt.exe")) {
        go build -o bin\verify_pjrt.exe .\scripts\verify_pjrt.go
    }
    .\bin\verify_pjrt.exe
    Write-Host "Note: GoMLX xla:cuda is NOT available on native Windows." -ForegroundColor Yellow
}

switch ($Mode) {
    "check" {
        Test-Cuda
        Test-XGBoostGPU
        Test-GoMLX-CPU
    }
    "xgb-gpu" {
        Test-Cuda
        $out = Join-Path $RepoRoot "testdata\model_gpu_native.json"
        python .\scripts\xgb_gpu_train_export.py --device cuda:0 --out $out
        Write-Host "Exported: $out" -ForegroundColor Green
        Write-Host 'Load in Go: io.LoadFromFile(path, io.DefaultLoadOptions())'
    }
    "leaves-cpu" {
        $env:GOMLX_BACKEND = "xla:cpu"
        go test -tags gomlx_train ./train/... -run "FitHistParallel|GPU" -count=1
        Write-Host "leaves training uses CPU multi-thread hist" -ForegroundColor Green
    }
}
