# 配置 leaves / GoMLX 使用已安装的 XLA PJRT CPU 插件。
# 用法（当前 PowerShell 会话）:
#   . .\scripts\setup_xla_env.ps1

$GoXlaDir = Join-Path $env:USERPROFILE "AppData\Local\go-xla"
$Dll = Join-Path $GoXlaDir "pjrt_c_api_cpu_plugin.dll"

if (-not (Test-Path $Dll)) {
    Write-Host "PJRT CPU 插件未找到: $Dll" -ForegroundColor Yellow
    Write-Host "请先运行: go build -o bin/install_pjrt.exe ./scripts/install_pjrt.go ; .\bin\install_pjrt.exe"
    return
}

$env:PJRT_PLUGIN_LIBRARY_PATH = $GoXlaDir
$env:GOMLX_BACKEND = "xla:cpu"
$env:TF_CPP_MIN_LOG_LEVEL = "3"

Write-Host "XLA PJRT 环境已设置:" -ForegroundColor Green
Write-Host "  PJRT_PLUGIN_LIBRARY_PATH = $GoXlaDir"
Write-Host "  GOMLX_BACKEND              = xla:cpu"
Write-Host ""
Write-Host "验证: go build -o bin/verify_pjrt.exe ./scripts/verify_pjrt.go ; .\bin\verify_pjrt.exe"
Write-Host "训练: go test -tags gomlx_train ./treebuilder/..."
