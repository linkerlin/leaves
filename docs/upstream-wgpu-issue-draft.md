# wgpu v0.30.x 计时异常上游报告（草稿，待提交）

> 环境：Windows 11 / leaves v2.7.1 / born v0.9.23（gogpu/wgpu v0.30.35，goffi v0.5.2）
> 复现仓库：github.com/linkerlin/leaves `scripts/born_upgrade_gate`

## 现象

1. **计时归零**：WebGPU 张量后端（Gather 链式批量树遍历）batch=64 实测 `NsPerOp≈0`
   （物理不可能；同代码 BornCPU 路径正常出数）。
2. **batch≥256 挂起**：更大批量计时永不返回（>120s），进程需强杀。
3. **驱动层 panic**（偶发）：`vkMapMemory returned null pointer (BUG-VK-001)`，
   buffer 上传阶段。
4. **GC 警告刷屏**：`WARN wgpu: Buffer released by GC (missing explicit Release)`
   ——born 侧 buffer 生命周期依赖 GC，未见显式 Release。

## 最小化方向

born `backend/webgpu`：小 tensor（Shape{64} float32）反复 `tensor.FromSlice` +
`Gather` + `.Data()` 往返，~40 轮内可复现 1/2。

## 疑点

- wgpu v0.29→v0.30（born v0.9.1→v0.9.23 同步升级）后出现；v0.29 未测（历史版本计时口径不可考）。
- CI（WARP 软件设备）表现为 `DXGI_ERROR_DEVICE_REMOVED`；本机（RTX 物理 GPU）为归零/挂起。

## 诉求

确认 v0.30.x 在 Windows 上 map/readback 语义是否有已知回归；born 侧是否需要显式
Release/submit 掩护。
