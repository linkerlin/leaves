# leaves v2.5.1

**主题**：CI 修复——自 v2.2.0 起的慢性红灯收口。

## Highlights

1. **CI wasm 构建**：`examples/wasm` 带 `//go:build js` 约束，构建步骤补 `GOOS=js GOARCH=wasm`（原报 "build constraints exclude all Go files"）。
2. **CI lint job**：golangci-lint-action v6 → v7（v6 不支持 golangci-lint v2；自 v2.1.6 升级 golangci-lint v2 起该 job 恒挂）。lint 以 `GOOS=windows` 分析——加速面 helper 仅被 `//go:build windows` 的 GPU 文件调用，linux 构建会误判 unused（升 v7 后才首次暴露）。
3. **CI Windows GPU panic**：runner 仅有 WARP 软件设备——可用性探测通过但运行时 `DXGI_ERROR_DEVICE_REMOVED` panic。新增 **`LEAVES_BORN_GPU=0|off|false`** 环境变量（Windows）：强制 `tree.BornWebGPUAvailable()` 返回 false，训练（`treebuilder` hist）与推理（`tree` Engine）的 WebGPU 路径全部回落 CPU；CI test job 已设置。见 [`docs/backend-auto.md`](backend-auto.md)。
4. **CRLF 测试解析**：`model` 包 contrib 测试在 CI Windows（autocrlf 检出）下解析 `"-0.670\r"` 失败 → 解析前 TrimSpace。
5. **runs.jsonl `elapsed_ms` 契约收紧**：账本行必带 `elapsed_ms`（`0` 表示 <1ms）——原 omitempty 在 sub-ms 训练时丢字段。

## Usage

```powershell
# 无真实 GPU / WARP 环境强制 CPU
$env:LEAVES_BORN_GPU = "0"
```

## Verification

- 本地（真实 WebGPU）：`LEAVES_BORN_GPU=0` 下 `go test ./tree ./model -count=1` 绿（knob 生效）。
- 本地 wasm 交叉编译：`GOOS=js GOARCH=wasm go build ./examples/wasm` 通过。
- CI 全 job 绿（本次发布起）。

## Compatibility

- **Breaking**：无。`LEAVES_BORN_GPU` 为新增 opt-out 开关，默认行为不变（真实 GPU 环境仍自动启用）。
