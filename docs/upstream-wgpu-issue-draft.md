# wgpu v0.30.x 计时异常上游报告（草稿，待提交）

> 环境：Windows 11 / leaves v2.7.2+ / born v0.9.23（gogpu/wgpu v0.30.35）
> 最小复现：`go run ./scripts/wgpu_repro`（Windows；`LEAVES_BORN_GPU=0` 应 exit 2）
> 完整模型计时：`go run ./scripts/born_upgrade_gate testdata/lg_breast_cancer.txt`

## 现象

1. **计时归零**：WebGPU 张量后端（Gather 链式批量树遍历）batch=64 实测 `NsPerOp≈0`
   （物理不可能；同代码 BornCPU 路径正常出数）。
2. **batch≥256 挂起**：更大批量计时永不返回（>120s），进程需强杀。
3. **驱动层 panic**（偶发）：`vkMapMemory returned null pointer (BUG-VK-001)`，
   buffer 上传阶段。
4. **GC 警告刷屏**：`WARN wgpu: Buffer released by GC (missing explicit Release)`
   ——born 侧 buffer 生命周期依赖 GC，未见显式 Release。

## 最小化方向

`scripts/wgpu_repro`：Shape{64} float32 反复 `tensor.FromSlice` + `Gather` + `.Data()`，
默认 80 轮。参考机曾在 ~40 轮内出现计时归零或挂起。

## 疑点

- wgpu v0.29→v0.30（born v0.9.1→v0.9.23 同步升级）后出现；v0.29 未测（历史版本计时口径不可考）。
- CI（WARP 软件设备）表现为 `DXGI_ERROR_DEVICE_REMOVED`；本机（RTX 物理 GPU）为归零/挂起。

## 诉求

确认 v0.30.x 在 Windows 上 map/readback 语义是否有已知回归；born 侧是否需要显式
Release/submit 掩护。
