# Changelog

本文件记录面向用户的版本变更。格式参考 [Keep a Changelog](https://keepachangelog.com/)。  
发版前勾选 [`docs/release-checklist.md`](docs/release-checklist.md)。

## [Unreleased]

（tag 之后在此累积下一次变更。）

## [2.1.0] - 2026-07-10

> **主题**：Agentic 闭环契约收口 + 库线平台化（扩展点 / BackendAuto 2.0 / 互操作等级 / 发版治理）  
> **Release 正文**：[`docs/release-notes-v2.1.0.md`](docs/release-notes-v2.1.0.md)

### Added

- **Agentic CLI 闭环**（`cmd/leaves`）：`sniff|train|eval|predict|inspect|explain|publish`
  - 完备 `params` 账本、`--save-best` 默认、`--from-run` 复现
  - `--error-format json`、`manifest.reproduce`、`--emit-repro-script`
  - sniff `data_quality`、`--na-policy error|skip-row`
- **扩展点**：objective / metric 全量 `Register`；[`docs/extension-points.md`](docs/extension-points.md)
- **BackendAuto 2.0**：batch≥64→BornCPU、≥256+GPU→BornGPU、稀疏/小批 Native；[`docs/backend-auto.md`](docs/backend-auto.md)
- **互操作等级**：稳定 / 实验 / 占位；`*io.LoadError` 可操作 hint；[`docs/interop-matrix.md`](docs/interop-matrix.md)
- **发版治理**：[`docs/api-surface.md`](docs/api-surface.md)、[`docs/versioning.md`](docs/versioning.md)、[`docs/release-checklist.md`](docs/release-checklist.md)
- `tree.BenchRecord` 统一 bench JSONL；`SelectBackendExplained` 可解释选型

### Changed

- `DefaultLoadOptions` 默认 `AutoTransform: true`（见 [NOTES.md](NOTES.md)）
- BackendAuto：大 batch 无 GPU 时选 BornCPU（不再静默 Native）
- NOTES 瘦身为仍有效兼容注记；路线图 v5.4 / 演进方案 v1.4 收口

### Fixed

- sniff `n_features` 与 `feature_names` 一致
- 早停路径默认保存 best_round 模型（`model_round`）
- README 中英文互链（`README.md` ↔ `README.en.md`）

### Documentation

- skills/leaves-autotrain + Cursor 镜像；examples/autotrain walkthrough 刷新

### CI / 质量

- 本地 `go test ./... -count=1` 已通过（发版前建议再跑 CI 三 OS）
- 门禁：`docs` 文档存在性、CLI 契约、BackendAuto 决策表、io 支持等级

---

## 打 tag

```powershell
# 工作区已提交后：
git tag -a v2.1.0 -m "leaves v2.1.0: Agentic CLI + platform hardening"
git push origin v2.1.0
# GitHub Release 正文粘贴 docs/release-notes-v2.1.0.md
```

---

## 更早版本

历史能力基线（P0–T5 训练完备、Born 迁移、量化/WASM 等）见 git history 与 [`TODO.md`](TODO.md) 存档。
