# leaves v2.5.5

**主题**：module zip 瘦身——21.8 MB → 约 5 MB。

## Highlights

1. **根因**：Go module zip 由 **git 追踪文件** 构成——任何被提交的产物文件都会被每个 `go get github.com/linkerlin/leaves/v2` 用户下载。
2. **移除**（加入 `.gitignore`）：
   - `bin/install_pjrt.exe` + `bin/verify_pjrt.exe`：~36 MB 原始体积的孤儿实验二进制（全仓零引用；pjrt 相关历史实验遗留）。
   - `examples/wasm/leaves.wasm`：3.5 MB 可重建产物——README 与 CI 均现场构建；wasm 体积门禁（`TestWasmBinarySizeGate`）在临时目录新构建测量，不依赖该文件。
   - `.chong/`：另一 Agent 工具的工作记忆/事件日志（11 个文件，与库无关）。
3. **保留**：`testdata/`（回归矩阵 `docs/testdata-matrix.md` 必需）、`logo.png`。

## Verification

- `rg pjrt` 全仓零引用（删前安全检查）；无代码引用 `examples/wasm/leaves.wasm` 路径。
- `go build ./...` + `go test ./docs ./cmd/leaves -count=1` 绿。
- fresh clone（v2.5.4）冒烟全通过后才发现本问题；v2.5.5 发布后以 module zip 实测大小复核。

## Compatibility

- **Breaking**：无。源码零变更；仅分发体积变化（变小）。`go run ./examples/wasm` 类路径照常（README 指导构建）。
