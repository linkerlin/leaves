# leaves v2.7.0

> **主题**：ONNX Classifier 子集 · BackendAuto profiling 接线 · 独立 serving 产品仓 · lessons 可检索记忆库 · release 自动批准  
> **日期**：2026-08-22

## Highlights

1. **ONNX TreeEnsembleClassifier 子集**：GBRT 分类器经 ONNX（onnxmltools 标准产物）
   现可走标准 `LoadFromFile` 路径——SUM/AVERAGE × NONE/SOFTMAX/LOGISTIC（二类）。
   通用图（30+ 算子）仍走 `LoadOnnxGraph`（v2.4）。
2. **BackendAuto 二轮 profiling 接线**：`LEAVES_BACKEND_PROFILE=1` 时以实测 ns/op
   替代阈值决策（形状类缓存，首调用延迟只付一次）；默认决策表 2.0 不变。
3. **独立 serving 产品仓**：[linkerlin/leaves-serving](https://github.com/linkerlin/leaves-serving)
   ——独立 module / 3-OS CI / Docker distroless；仓内模板 require 同步修复。
4. **lessons 可检索记忆库**：`leaves lessons add|search|list`（`~/.leaves/lessons.jsonl`）；
   SKILL §4.6 双层记忆（全局库 + 项目镜像）；闭环步骤 -1/8 接 CLI。
5. **release 自动批准**：`AutoApprovePolicy` + `Machine.AutoApprove`——策略化
   candidate→approved，审计等价人工；warn 门禁可配置拒绝；人工批准仍默认。

## Recommended API

- Infer: `LoadFromFile` + `DefaultLoadOptions`（.onnx 分类器/回归器子集自动识别）
- Train: `NewLearner` / `LoadDataAuto` / CLI `leaves train`
- Memory: `leaves lessons add|search|list`
- Recsys release: `release.AutoApprovePolicy` / `Machine.AutoApprove`
- See `docs/api-surface.md`

## Compatibility

- Breaking: none
- 默认行为不变：BackendAuto 决策表、人工 Approve、`SelectBackendExplained` 规则码
- `LEAVES_BACKEND_PROFILE` / `LEAVES_LESSONS_PATH` 均为 opt-in 新开关

## CI

- test（3 OS）/ lint / race / wasm / bench-gate；fuzz workflow（每周）
- 独立仓 leaves-serving CI 3 OS 绿

## Docs

- `skills/leaves-autotrain/SKILL.md` §4.6 + 速查卡；`cli.md` lessons 节（镜像同步）
- `docs/backend-auto.md` profiling env 接线；`docs/recsys-loop.md` §8 自动批准
- CHANGELOG Unreleased → [2.7.0]
