# leaves Agent 自动训练 demo（零准备）

一个小型回归数据集，用于**首次体验** SKILL 驱动的全自动闭环。无需准备数据，照抄下面的命令即可跑通「嗅探→训练→评估→发布」。

> 背后的方法论文档：[`skills/leaves-autotrain/SKILL.md`](../../skills/leaves-autotrain/SKILL.md)
> CLI 全参考：[`skills/leaves-autotrain/cli.md`](../../skills/leaves-autotrain/cli.md)

## 数据

- `data/train.csv`（120 行，3 特征 + 末列 label，`y ≈ 2·x0 − 1.5·x1 + 0.8·x2 + 微噪`）
- `data/holdout.csv`（30 行，独立验证）

## 一键闭环（PowerShell）

```powershell
# 仓库根目录执行。把 $RUNS 指向任意工作区。
$RUNS = "examples/autotrain/run"

# 1) 嗅探：自动识别任务类型（这里会推荐 reg:squarederror / rmse）
go run ./cmd/leaves sniff --data examples/autotrain/data/train.csv --metrics $RUNS/sniff.json

# 2) 基线训练：5 折 CV，出模型 + metrics.json，并记入账本
go run ./cmd/leaves train `
  --data examples/autotrain/data/train.csv `
  --objective reg:squarederror --eval-metric rmse `
  --cv 5 --rounds 40 --depth 4 --lr 0.2 `
  --out-model $RUNS/m.leaves.json --metrics $RUNS/metrics.json `
  --runs $RUNS/runs.jsonl --tag baseline

# 3) 独立 holdout 验收
go run ./cmd/leaves eval `
  --model $RUNS/m.leaves.json `
  --data examples/autotrain/data/holdout.csv `
  --eval-metric rmse --metrics $RUNS/holdout.json

# 4) 发布：leaves.json + XGB 导出 + int8 量化侧车 + manifest
go run ./cmd/leaves publish `
  --model $RUNS/m.leaves.json --out-dir $RUNS/release `
  --version 1.0.0 --export-xgb --quantize `
  --data examples/autotrain/data/train.csv --metrics $RUNS/metrics.json
```

跑完后 `$RUNS/release/` 下应有 `model.leaves.json`、`model.quant.json`（量化侧车，可被 `predict` 重建）、`model.xgb.json`、`quantize_report.json`、`manifest.json`。

## 让 Agent 继续优化

把上面的命令交给任意 Agent（读 [`SKILL.md`](../../skills/leaves-autotrain/SKILL.md)），它会：

1. 读 `metrics.json` 的 `value`/`cv_mean` 与账本的 `fold_metrics`/`n_trees`/`elapsed_ms`，按 SKILL §4.5 演化搜索协议（HoF 选父 → 反射式定向变异 → 筛选晋级）选下一组超参；
2. 再 `train` 一轮、`--tag p:<父tag>+<变异>`（谱系）、追加进同一个 `runs.jsonl`；
3. 比较账本取最优，达 §五 收敛判据后 `publish`。
4. 定稿复现可用：`train --data ... --from-run $RUNS/runs.jsonl --tag <最优> --out-model ...`（CLI flag 覆盖账本）。

> 提示：训练的加速日志打到 stderr，指标只进 `--metrics` 文件或 stdout（JSON）。Agent 读文件即可，无需解析日志。  
> **路径纪律**：`--out-model` 与 `--metrics` 必须是不同路径（同一路径会被 CLI 拒绝，因 metrics 会覆盖模型）。
