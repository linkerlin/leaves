# leaves v2.2.0

**主题**：向量叶 `multi_output_tree` 训练（multi-target 一棵树、叶子为向量）。

## Highlights

1. **向量叶训练落地**（原列「明确不做」）：XGBoost `multi_output_tree`（`size_leaf_vector=num_groups`）—— 每轮长**一棵**树，叶子为 `numGroups` 维向量；分裂增益跨输出组求和，叶权重逐类 `-G_c/(H_c+λ)`。此前仅推理/加载支持向量叶，训练只有 `one_output_per_tree`。
2. **统一 k 维 treebuilder**：grad/hess `[n*k]`、`splitGain` 跨类求和、leaf-major `LeafValue`。`k=1` 退化为标量 —— **现有训练路径零回归**（全量 golden 套件验证）。
3. **exact + hist 双路径**都支持向量叶；Born/WebGPU 增益扫描对 `k>1` 回退纯 CPU（标量路径仍享加速）。

## Usage

```go
learner, _ := train.NewLearner(train.Config{
    Objective:       "multi:softprob", // 或 reg:squarederror + NumTarget>1
    NumClass:        5,                 // 多输出
    MultiOutputTree: true,              // ← 向量叶（默认 false = one_output_per_tree）
    NumRound:        50,
    TreeMethod:      "hist",
    // ...
})
learner.Fit(dm)
```

`MultiOutputTree=true` 时：每轮一棵 `OutputDim=numGroups` 的树；推理按 `OutputDim` scatter 到所有输出组（与加载的 XGBoost 向量叶模型一致）。`NewLearner` 校验：单输出目标开此选项会报错。

## Design notes

- **为何统一而非复制**：把 treebuilder 数学参数化为 `k`，而非为向量叶复制一套并行代码 —— 更少代码、单一代码路径可测，`k=1` 由现有 golden 套件保证不变。
- **accel 取舍**：Born/WebGPU 增益扫描当前为标量 kernel；向量叶（`k>1`）回退纯 CPU gain scan。正确优先，性能留信号再开（k=1 标量训练仍享 Born/WebGPU 加速）。

## Verification

- 确定性手算测试（4 样本×k=2：分裂点、leaf 向量、OutputDim 逐项断言）
- 端到端多目标训练：每棵树 OutputDim=2、收敛（rmse<0.35）、leaves.json 往返保留 OutputDim
- 单输出 + MultiOutputTree → NewLearner 报错
- 全量回归：13 包 `-short` 全绿、6-linter 门禁 0 issues、race detector 无竞争

## Compatibility

- **Breaking**：无。`MultiOutputTree` 默认 `false`，现有 `one_output_per_tree` 行为完全不变。
- 向量叶**推理/加载/保存**此前已支持，本版本补齐**训练**端到端闭环。

## Recommended API

- Infer：`LoadFromFile` + `DefaultLoadOptions`
- Train：`NewLearner`（多输出可开 `MultiOutputTree`）/ `LoadDataAuto` / CLI `leaves train`
- 详见 [`docs/api-surface.md`](api-surface.md)
