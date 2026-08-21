# leaves

[English](README.en.md) | **中文**

[![版本](https://img.shields.io/badge/版本-v2.6.0-blue.svg)](https://github.com/linkerlin/leaves/releases/tag/v2.6.0)
[![CI](https://github.com/linkerlin/leaves/actions/workflows/ci.yml/badge.svg)](https://github.com/linkerlin/leaves/actions/workflows/ci.yml)
[![Go 文档](https://pkg.go.dev/badge/github.com/linkerlin/leaves/v2.svg)](https://pkg.go.dev/github.com/linkerlin/leaves/v2)
[![许可证](https://img.shields.io/badge/许可证-MIT-green.svg)](LICENSE.md)

![Logo](logo.png)

## 简介

`leaves` 是一个**纯 Go** 写的推荐系统用排序模型**训练与推理**库。能直接加载 XGBoost / LightGBM / scikit-learn 模型，也能自训练（hist / exact、排序、生存、tweedie 等），全程不需要任何 C 绑定。

**推荐入口**

- 推理：`leaves.LoadFromFile`（默认开启 `AutoTransform`），或者沿用旧的 `leaves.LGEnsembleFromFile` / `leaves.XGEnsembleFromFile`。
- 训练：`leaves.NewLearner` / `train.NewLearner`、`leaves.LoadDataAuto` / `data.FromFileAuto`（带内容嗅探），以及便利函数 `leaves.NewLearnerFromModelAndData` —— 它能从参考模型里反推出 objective。

回归矩阵见 [docs/testdata-matrix.md](docs/testdata-matrix.md)；路线图 [演进计划.md](演进计划.md)（v5.4）；Agent 契约 [演进方案.md](演进方案.md)（v2.2）；扩展 objective/metric 见 [docs/extension-points.md](docs/extension-points.md)；backlog [TODO.md](TODO.md)。

## 特性

### 通用

- 并行批预测（dense / CSR）
- 支持 sigmoid、softmax 变换；`DefaultLoadOptions()` 默认开启 **AutoTransform**
- 可输出决策树的叶节点索引
- **训练数据内容嗅探**（LIBSVM、排序 TSV、CSV·TSV —— 扩展名只是回退依据）

### [LightGBM](https://github.com/microsoft/LightGBM)

- 兼容 `text` 与 `JSON` 两种模型格式
- 支持 `gbdt`、`rf`（随机森林）、`dart` 三类模型
- 支持多分类预测
- 对类别特征（独热分割等）有专门的决策规则优化
- 纯预测场景下的快速路径

### [XGBoost](https://github.com/dmlc/xgboost)

- 支持二进制、JSON、[UBJSON](https://github.com/jmckaskill/ubjson) 三种模型格式
- `gbtree` / `gblinear` / `dart` 三种 booster；3.x 嵌套 DART `gbtree.model`
- 多分类预测、多输出向量叶、XGBoost 原生 categorical（`model.cats`）并支持 round-trip 导出
- 缺失值（`nan`）支持
- `reg:gamma` / `count:poisson` / `reg:tweedie` 自动加载变换（Exp + `base_score` 对数转换）

### [scikit-learn](https://github.com/scikit-learn/scikit-learn)（实验性）

- 支持 `pickle` 协议 `0`
- 兼容 `sklearn.ensemble.GradientBoostingClassifier`

### 训练（v2.x，里程碑 T1–T5）

- `gbtree` / `dart` / `gblinear`；`hist` / `exact` / `gpu_hist`（`auto` 模式按行数自动选）
- 目标函数：回归、二分类 / 多分类、gamma、poisson、tweedie、`survival:cox` / `survival:aft`，以及排序（`rank:ndcg` / `rank:pairwise` / `rank:listwise`）
- 交叉验证、早停、checkpoint、续训、单调约束、`max_leaves`（lossguide）、`BatchedMatrix`（外存 DMatrix）
- 模型导出：原生 `leaves.json`、**XGBoost 3.x JSON**

### 部署与观测

- int8 阈值量化（带 parity 门禁）
- `Ensemble.Reload` 线上热更新；按 batch 维度的推理 profiling
- [WASM demo](examples/wasm/README.md)、[HTTP embed](examples/http/README.md)、[serving 模板](examples/serving-template/README.md)、[train-from-model](examples/train_from_model/README.md)、[扩展 objective/metric](examples/extension/README.md)、[多目标回归](examples/multitarget/README.md)
- [MovieLens Ranker + Agent/MCP 教程](demos/movielens/TUTORIAL.md)（精排全流程案例）

## 安装

```sh
go install github.com/linkerlin/leaves/v2/cmd/leaves@latest
```

模块路径：`github.com/linkerlin/leaves/v2`（Go 1.26+）。张量与 GPU 加速由 [github.com/born-ml/born](https://github.com/born-ml/born) 提供。

## 快速上手

最简单的方式：用 `leaves.LoadFromFile` 加载模型，然后调 `PredictSingle`。

```go
package main

import (
	"fmt"

	"github.com/linkerlin/leaves/v2/v2"
)

func main() {
	m, err := leaves.LoadFromFile("lightgbm_model.txt", leaves.DefaultLoadOptions())
	if err != nil {
		panic(err)
	}

	fvals := []float64{1.0, 2.0, 3.0}
	p, err := m.PredictSingle(fvals, 0)
	if err != nil {
		panic(err)
	}
	fmt.Printf("对 %v 的预测: %f\n", fvals, p)
}
```

旧 API 仍兼容：`leaves.LGEnsembleFromFile(path, loadTransformation)` 和 `leaves.XGEnsembleFromFile(...)`，行为不变。

### 格式自动识别与支持等级

`io.LoadFromFile` / `DetectFormat` 看扩展名与文件头。完整等级表见 **[docs/interop-matrix.md](docs/interop-matrix.md)**（稳定 / 实验 / 占位）。

| 格式 | 等级 | 扩展名 / 探测 | 说明 |
|------|------|---------------|------|
| leaves.json | **稳定** | `leaves_version` | 训练主产物；推荐部署 |
| XGBoost JSON | **稳定** | `.json` + `learner` | 3.x 默认；`base_score`→margin |
| XGBoost UBJSON | **稳定** | `.ubj` | 与 JSON 预测一致 |
| XGBoost binary | **稳定** | `binf` / header | 经典 Booster；优先改用 JSON |
| LightGBM text/JSON | **稳定** | `tree=` / `tree_info` | text 与 JSON |
| scikit-learn | **实验** | `.pkl` / `.joblib` | 窄协议；生产请转 XGB/leaves JSON |
| ONNX | **实验** | `.onnx` | TreeEnsembleRegressor 子集；复杂图请先转 JSON/leaves |

加载失败返回 `*io.LoadError`（含 `hint:` 下一步）。数值表误用 `.txt` 会提示改用 `data.FromFile`。

```go
import "github.com/linkerlin/leaves/v2/io"

m, err := io.LoadFromFile("model.ubj", io.DefaultLoadOptions()) // AutoTransform = true

m, err = io.LoadFromFile("model.ubj", &io.LoadOptions{
	AutoTransform:      false, // 输出原始 margin
	LoadTransformation: true,  // 或者对所有目标强制启用变换
	Backend:            io.BackendAuto,
})
```

当 `AutoTransform` 与 `LoadTransformation` 都为 `false` 时，`OutputValue` 和 `OutputMargin` 都是原始 margin。`contrib` / SHAP 始终在 margin 空间，与 `AutoTransform` 无关。

## 训练 API

```go
dm, _ := leaves.LoadDataAuto("train.tsv") // 内容嗅探

learner, _ := leaves.NewLearner(leaves.TrainConfig{
	Objective: leaves.TrainObjectiveBinaryLogistic,
	NumRound:  50,
})
_ = learner.Fit(dm)
_ = learner.Save("out.leaves.json")
```

### 外存 DMatrix

```go
bm, _ := data.SplitDense(dense, batchRows) // 也可以 data.NewBatchedMatrix(...)
learner, _ := train.FitExternal(train.Config{
	Objective:     train.ObjectiveBinaryLogistic,
	TreeMethod:    train.TreeMethodHist,
	HistBinPolicy: "global",
	NumRound:      10,
}, bm)
```

### Checkpoint 续训

```go
learner, _ := train.ResumeFit("ckpt.json", train.Config{
	Objective:  train.ObjectiveSquaredError,
	NumRound:   20, // 在 checkpoint 完成的轮次之上继续
	TreeMethod: train.TreeMethodExact,
}, dm)
score, _ := learner.Eval(dm)
```

### 训练数据 —— 自动嗅探

`data.FromFileAuto` / `leaves.LoadDataAuto` 会先读一段样本来选 loader，扩展名只是兜底。

| 嗅探结果        | 典型内容                | Loader           |
| --------------- | ----------------------- | ---------------- |
| LIBSVM          | `label 1:0.5 3:1.2`     | `FromLIBSVM`     |
| 排序 TSV        | `qid label feat...`     | `LoadRankingTSV` |
| 末列为 label 的 TSV | 无表头，末列是 label | `LoadDenseTSV`   |
| CSV             | 有表头或启发式 label 列 | `FromCSV`        |

### 从参考模型反推 objective

```go
learner, _ := leaves.NewLearnerFromModelAndData(
	"reference_model.json", // 只读 objective（XGB JSON / UBJ / leaves.json）
	"train.tsv",            // LoadDataAuto 嗅探
	leaves.TrainConfig{NumRound: 10, TreeMethod: leaves.TrainTreeMethodExact},
	data.DefaultFileLoadOptions(),
)
_ = learner.Save("out.leaves.json")
```

```bash
go run ./examples/train_from_model/ -model testdata/xgboost_smoke.json -data testdata/breast_cancer_train.tsv
```

### `survival:aft` 区间删失

```go
d, _ := data.NewDense(vals, n, nfeat, placeholderLabels, nil)
dm, _ := data.NewAFTDense(d, []data.AFTInterval{
	{1.5, 1.5},              // 事件
	{2, math.Inf(1)},        // 右删失
	{0, 4},                  // 左删失
	{1, 6},                  // 区间删失
})
learner, _ := train.NewLearner(train.Config{Objective: train.ObjectiveSurvivalAFT, ...})
_ = learner.Fit(dm)
```

### 单调约束

```go
learner, _ := train.NewLearner(train.Config{
	Objective:           "reg:squarederror",
	MonotoneConstraints: []int{1, 0, -1}, // 特征 0 递增，特征 2 递减
	TreeMethod:          train.TreeMethodHist,
})
```

### 训练加速栈（T4+）

hist / `gpu_hist` 路径上的加速分层（日志前缀 `[leaves/train] accel:`）：

| 阶段        | 实现                                              | 默认 `auto`（≥3 万行 + WebGPU） | `LEAVES_TRAIN_ACCEL=webgpu` |
| ----------- | ------------------------------------------------- | ------------------------------- | --------------------------- |
| 全局分箱    | 训练期每特征一次切点 + 行级 bin 缓存              | ✅                              | ✅                           |
| 直方图累加  | CPU `rowBin` / WebGPU `SelectAdd` 批量            | `gpu_hist`                      | GPU batch                    |
| 增益扫描    | 纯 CPU / 批量 WebGPU gain                        | 批量 GPU（GPU 路径）             | 批量 GPU                     |
| margin 预测 | 逐行 CPU / Born GPU `PredictDense`                | `PredictMargins` ≥256 行        | ≥64 行                       |

```go
learner, _ := train.NewLearner(train.Config{
	Objective:     train.ObjectiveRankNDCG,
	TreeMethod:    train.TreeMethodGPUHist,
	AccelMode:     train.AccelModeWebGPU, // 也可走环境变量 LEAVES_TRAIN_ACCEL
	NumThreads:    4,
	HistBinPolicy: "global", // 默认；"per_node" 恢复旧的 per-node 切点
	NumRound:      50,
})
_ = learner.Fit(dm)
```

环境变量覆盖：`LEAVES_TRAIN_ACCEL=auto|webgpu|born_cpu|cpu`。

## 计算底座 —— [Born](https://github.com/born-ml/born)

推理与训练加速都构建在 [Born](https://github.com/born-ml/born) 之上（CPU SIMD + WebGPU）。`NativeEngine` 是 golden 基准；`BackendAuto`（**2.0**）按 workload 派发 BornCPU / BornGPU。完整决策表：[docs/backend-auto.md](docs/backend-auto.md)。

```go
m, _ := leaves.LoadFromFile("model.json", &io.LoadOptions{
	Backend:  io.BackendAuto,
	Workload: tree.WorkloadHint{BatchSize: 256, HasGPU: true},
})
// 可解释选型：tree.SelectBackendExplained(caps, hint) → Rule / Reason
```

parity 门禁 `TestBornParityFormatMatrix` 覆盖 LGB text/JSON、XGB bin/json/ubj、scikit-learn pickle × batch `{1, 16, 256}` × `BornCPU` / `BornGPU`，容差 `1e-5`（相对 Native）。

### 后端选型速查（BackendAuto 2.0）

| 场景 | Auto 结果 | 说明 |
|------|-----------|------|
| 在线单条 / batch &lt; 64 | **Native** | 延迟优先；Born 单条更慢 |
| 批量 ≥ 64，纯数值树 | **BornCPU** | SIMD 批推理 |
| 批量 ≥ 256 + HasGPU（Windows WebGPU） | **BornGPU** | 不可用则回落 BornCPU |
| WASM + 数值树 | **BornCPU**（支持时） | 无 GPU；cat-small → Native |
| 稀疏 `SparseDensity∈(0,0.15)` / cat-small | **Native** | Born 未优化 / 不支持 |

## Tree SHAP 与可解释性

`m.Explain()` 暴露的是 Lundberg 快速 Tree SHAP（带 `SumHess` 权重覆盖，`tree_path_dependent`）；SHAP 值在 margin 空间，与 `OutputMargin` / `predict.OutputContribution` 保持一致。`DefaultLoadOptions()` 对 logistic 模型返回**概率**；contrib / SHAP 仍在 margin 空间。

```go
import (
	"github.com/linkerlin/leaves/v2/v2"
	"github.com/linkerlin/leaves/v2/explain"
)

m, _ := leaves.LoadFromFile("model.json", leaves.DefaultLoadOptions())

x := [][]float64{{1.0, 2.0, 3.0}}
contrib, _ := m.Explain().TreeSHAP(x)        // margin 空间 SHAP
inter, _ := m.Explain().InteractionSHAP(x)   // 交互值矩阵
mc, _ := m.Explain().TreeSHAPMulticlass(x)   // 多类：[sample][feature][class]
base := m.Explain().ExpectedValue()          // 背景基线（全零特征）

imp := m.Explain().Importance(explain.ImportanceGain, nil)
text := m.Explain().DumpText(nil)
dot := m.Explain().DumpDOT(nil)
```

统一出口是 [`predict.Request`](predict/request.go)：

```go
import "github.com/linkerlin/leaves/v2/predict"

nf := m.NFeatures()
nRows := 1

// 输出值（logistic 已变换，squarederror 仍为原始）
vals := make([]float64, nRows*m.NOutputGroups())
_ = m.PredictWithRequest(predict.Request{
	Matrix: predict.DenseMatrix{Values: flat, Rows: nRows, Cols: nf},
	Output: predict.OutputValue,
}, vals)

// 原始 margin（sigmoid / softmax 之前）
margins := make([]float64, nRows*m.NRawOutputGroups())
_ = m.PredictWithRequest(predict.Request{
	Matrix: predict.DenseMatrix{Values: flat, Rows: nRows, Cols: nf},
	Output: predict.OutputMargin,
}, margins)

// 叶节点索引（每棵树一个 int32）
nTrees := m.NTrees()
leaves := make([]int32, nRows*nTrees)
_ = m.PredictWithRequest(predict.Request{
	Matrix: predict.DenseMatrix{Values: flat, Rows: nRows, Cols: nf},
	Output: predict.OutputLeaf,
}, leaves)

// SHAP 贡献（margin 空间；二分类 [sample][feature+bias]）
out := make([]float64, (nf+1)*nRows)
_ = m.PredictWithRequest(predict.Request{
	Matrix: predict.DenseMatrix{Values: flat, Rows: nRows, Cols: nf},
	Output: predict.OutputContribution, // 也可 OutputApproxContribution / OutputInteraction
}, out)
```

`bias` 语义：末列是**全零特征处的背景 margin**，与 XGBoost `pred_contribs` 的可加性约定一致（逐元素分解可能不同）。

## 评估指标 —— `metrics/`

内置 RMSE / MAE / AUC / LogLoss / MAPE / RMSLE / NDCG@k / MAP，命名与 XGBoost `eval_metric` 对齐：

```go
import "github.com/linkerlin/leaves/v2/metrics"

rmse, _ := metrics.RMSE{}.Evaluate(yTrue, yPred)
m, _ := metrics.Resolve("ndcg@5", metrics.Options{Groups: []int{10, 10}})
ndcg, _ := m.Evaluate(yTrue, yPred)
```

排序指标需要 `data.GroupedMatrix` 提供 `Groups()`。

## 排序学习

XGBoost 兼容的 LambdaMART，再加上原生的 listwise 头：

| 目标            | 家族                                       | 说明                                                                       |
| --------------- | ------------------------------------------ | -------------------------------------------------------------------------- |
| `rank:ndcg`     | LambdaMART（XGBoost 兼容）                  | pairwise + \|ΔNDCG\| 缩放，直接对齐排序指标                                |
| `rank:pairwise` | RankNet pairwise                            | 默认 **top-k**（k=32）+ `lambdarank_normalization`（对标 XGBoost）          |
| `rank:listwise` | **ListNet softmax CE**（leaves 原生）       | `q ∝ exp(label)`、`p = softmax(pred)`，纯 listwise 交叉熵                  |

```go
import (
	"github.com/linkerlin/leaves/v2/data"
	"github.com/linkerlin/leaves/v2/train"
)

dm, _ := data.LoadRankingTSV("rank_train.tsv", "\t") // qid label feat1 feat2 ...

learner, _ := train.NewLearner(train.Config{
	Objective:    train.ObjectiveRankNDCG, // 或 ObjectiveRankPairwise / ObjectiveRankListwise
	NumRound:     40,
	MaxDepth:     4,
	LearningRate: 0.1,
	NDCGK:        10,
	EvalMetric:   "ndcg@10",
})
_ = learner.Fit(dm)
```

端到端 MovieLens 100K demo（训练 → 保存 → Top-K 推荐）：[`demos/movielens/README.md`](demos/movielens/README.md)。

```bash
cd testdata && python gen_rank_movielens.py
go run ./demos/movielens/cmd/train
go run ./demos/movielens/cmd/recommend -group 0 -topk 10
go test ./train/... -run TestRankMovieLens -v
```

## 训练回调与学习率调度

```go
learner, _ := train.NewLearner(train.Config{
	Objective:    train.ObjectiveSquaredError,
	NumRound:     20,
	LearningRate: 0.3,
	LRScheduler:  train.ExponentialLRScheduler(0.95), // 每轮 ×0.95
	Callbacks: []train.TrainingCallback{
		train.FuncCallback(func(ctx *train.CallbackContext) error {
			// ctx.Round, ctx.LearningRate, ctx.TrainMetric, ctx.EvalMetric
			return nil
		}),
	},
})
```

内置调度器：`ExponentialLRScheduler(gamma)` 和 `StepLRScheduler(every, factor)`。配置 `EvalSet` + `EvalMetric` 时，`CallbackContext` 也会自动填充 `EvalMetric` / `EvalMetricOK`（不依赖 `EarlyStop`）。

## 推理 profiling 与热更新

```go
if ne, ok := m.Engine().(*tree.NativeEngine); ok {
	prof, err := tree.ProfileNativeDense(ne, vals, nrows, ncols, preds, 0)
	_ = prof.Elapsed
}

// 线上原子切换模型
_ = m.Reload("/path/to/new.model", io.DefaultLoadOptions())
```

`tree.ProfileWalkStats` 可以单独统计单样本的树遍历深度，不跑完整预测。

## int8 阈值量化

数值分裂阈值按特征做 int8 仿射量化（127 级），**叶子值保持 float64**，分类节点不量化。`quantize.Engine` 不支持 `PredictLeafIndices*`（只支持 margin 预测）。

```go
qf, _ := quantize.QuantizeForest(m.Forest(), quantize.Config{})
res, err := quantize.CheckParityWithGate(m.Forest(), qf, samples, 0, quantize.DefaultGate())

eng, _ := quantize.NewEngine(qf, nil, tree.TransformRaw, m.NOutputGroups())
model.NewEnsemble(eng) // 替换线上 Ensemble 引擎
```

## Agent 自动化（SKILL 驱动，无 MCP）

> **通用 Agentic 训练契约已达成**（见 [`演进方案.md`](演进方案.md) v2.2 DoD）；推荐生产闭环控制面（§十七 RC）亦已落地，见 [`docs/recsys-loop.md`](docs/recsys-loop.md)。
> 库整体 12 个月路线图见 [`演进计划.md`](演进计划.md) v5.4 — 二者互补，不互相替代。

Agent 通过 **SKILL + `leaves` CLI + metrics.json** 完成「训练→调参→发布」；**Agent 即优化器**（搜索逻辑在 SKILL，库不内置 HPO）。

### 三步让 Agent 帮你训练（用户视角）

1. **装 CLI**：`go install github.com/linkerlin/leaves/v2/cmd/leaves@latest`（或在仓库内用 `go run ./cmd/leaves ...`）。
2. **把技能给 Agent**：Cursor 用户开箱即得（`.cursor/skills`）；Claude Code 读 [`CLAUDE.md`](CLAUDE.md) → [`AGENTS.md`](AGENTS.md)；其他 Agent 让它读 [`skills/leaves-autotrain/SKILL.md`](skills/leaves-autotrain/SKILL.md)。
3. **对 Agent 说一句话**（无需懂 flag）：

```text
用 leaves-autotrain 技能，训练数据在 data/train.csv，
目标把 RMSE 降下来，收敛后发布到 release/v1。
```

Agent 会自动：嗅探数据定任务类型 → CV 基线 → 按 SKILL §4.5 演化搜索协议调参（runs.jsonl 记账、早停保最优轮）→ 收敛后 `publish` 出工件包（模型 + manifest + 复现命令）。全程只读 JSON，不写 Go 代码。

| 文档 | 用途 |
|------|------|
| [`演进方案.md`](演进方案.md) | Agentic 契约 / WP / 验收 DoD |
| [`skills/leaves-autotrain/SKILL.md`](skills/leaves-autotrain/SKILL.md) | 决策表与闭环策略 |
| [`skills/leaves-autotrain/cli.md`](skills/leaves-autotrain/cli.md) | flag + metrics schema |
| [`examples/autotrain/`](examples/autotrain/README.md) | 零准备 demo |

```
sniff → train(--cv --runs --tag [--early-stop 默认 save-best] [--from-run])
  → 读 metrics/runs → 决策表调参 → 收敛 → inspect → publish(--emit-repro-script)
```

边界：**本地工件包**（非 registry）；不内置 HPO；CSV 默认无缺失（`--na-policy skip-row` 可丢行，不做插补）。

```powershell
go run ./cmd/leaves --error-format json sniff --data train.csv
go run ./cmd/leaves train --data train.csv --objective reg:squarederror --cv 5 --runs runs.jsonl --tag baseline
go run ./cmd/leaves train --data train.csv --from-run runs.jsonl --tag baseline --out-model m.leaves.json --metrics metrics.json
go run ./cmd/leaves publish --model m.leaves.json --out-dir release/ --quantize --export-xgb --metrics metrics.json --emit-repro-script both
```
`publish --quantize` 会持久化 int8 量化侧车（`model.quant.json`）；`manifest.json` 含 `reproduce` 复现命令与文件 sha256。

### 推荐生产闭环控制面（§十七 RC）

`recsys/` 在离线四段（prep→召回→LTR→发牌）之上提供**可离线测试的控制面契约**（纯 Go，无运行时依赖）：数据快照/指纹（`recsys/contract`）、时间切分防泄漏（`recsys/split`）、三层离线门禁（`recsys/eval`）、决策/曝光/反馈账本（`recsys/ledger`）、归因回放（`recsys/replay`）、窗口监控（`recsys/monitor`）、发布状态机 + adapter 请求（`recsys/release`）。端到端演练见 `recsys/loop`。

- 指南：[`docs/recsys-loop.md`](docs/recsys-loop.md) · 剧本：[`skills/recsys-orchestrator/SKILL.md`](skills/recsys-orchestrator/SKILL.md) §十
- 边界：离线 `deal` ≠ 线上 decision；leaves 只产出推广/回滚**请求**（fake adapter 可测），不托管 registry/serving/在线学习

## 文档

| 文档 | 说明 |
|------|------|
| [godoc](https://pkg.go.dev/github.com/linkerlin/leaves/v2) | API 参考 |
| [docs/api-surface.md](docs/api-surface.md) | **推荐 / 兼容 / 实验** API 分层与迁移 |
| [docs/versioning.md](docs/versioning.md) | v2.x 允许改什么 |
| [docs/release-checklist.md](docs/release-checklist.md) | **v2.1 发版检查表** |
| [CHANGELOG.md](CHANGELOG.md) | 面向用户的变更记录（Unreleased → tag） |
| [docs/interop-matrix.md](docs/interop-matrix.md) | 格式支持等级 |
| [docs/backend-auto.md](docs/backend-auto.md) | BackendAuto 2.0 决策表 |
| [docs/extension-points.md](docs/extension-points.md) | 自定义 objective/metric |
| [演进计划.md](演进计划.md) | 库 12 个月路线（v5.4） |
| [演进方案.md](演进方案.md) | Agent 闭环契约（已达成）+ §十七 RC |
| [docs/recsys-loop.md](docs/recsys-loop.md) | 推荐生产闭环控制面（RC） |
| [TODO.md](TODO.md) | 可执行 backlog |
| [NOTES.md](NOTES.md) | 仍有效的兼容注记（AutoTransform 等） |
| [compatibility.md](compatibility.md) | 外部 GBRT 库正确性矩阵 |
| [AGENTS.md](AGENTS.md) | 项目规约 |
| [docs/testdata-matrix.md](docs/testdata-matrix.md) | 回归矩阵 |
| [docs/benchmark-baseline.md](docs/benchmark-baseline.md) | Bench 门禁 |
| [examples/](examples/) | autotrain / wasm / http / serving-template / train_from_model / extension / multitarget |

## 兼容性

- **API 分层与迁移**：[docs/api-surface.md](docs/api-surface.md)  
- **版本策略**：[docs/versioning.md](docs/versioning.md)  
- **行为注记**（AutoTransform 默认等）：[NOTES.md](NOTES.md)  
- **外部库版本矩阵**：[compatibility.md](compatibility.md)  

## 性能

单线程 / batch=1000，硬件：2017 款 15 寸 MacBook Pro（2.9 GHz Core i7，16 GB 2133 MHz LPDDR3）。C API 走 Python 绑定，批调用下绑定开销可忽略。`go test -bench` 驱动 leaves 一侧；数据准备流程见 [benchmark/](benchmark/) 和 [testdata/README.md](testdata/README.md)。

单线程：

| 测试用例                                                                                  | 特征数 | 树数 | 批量  | C API | _leaves_ |
| ---------------------------------------------------------------------------------------- | ------ | ---- | ----- | ----- | -------- |
| LightGBM [MS LTR](https://github.com/microsoft/LightGBM/blob/master/docs/Experiments.rst#comparison-experiment) | 137    | 500  | 1000  | 49 ms | 51 ms    |
| LightGBM [Higgs](https://github.com/microsoft/LightGBM/blob/master/docs/Experiments.rst#comparison-experiment)   | 28     | 500  | 1000  | 50 ms | 50 ms    |
| LightGBM KDD Cup 99*                                                                     | 41     | 1200 | 1000  | 70 ms | 85 ms    |
| XGBoost Higgs                                                                            | 28     | 500  | 1000  | 44 ms | 50 ms    |

四线程：

| 测试用例                                                                                  | 特征数 | 树数 | 批量  | C API | _leaves_ |
| ---------------------------------------------------------------------------------------- | ------ | ---- | ----- | ----- | -------- |
| LightGBM [MS LTR](https://github.com/microsoft/LightGBM/blob/master/docs/Experiments.rst#comparison-experiment) | 137    | 500  | 1000  | 14 ms | 14 ms    |
| LightGBM [Higgs](https://github.com/microsoft/LightGBM/blob/master/docs/Experiments.rst#comparison-experiment)   | 28     | 500  | 1000  | 14 ms | 14 ms    |
| LightGBM KDD Cup 99*                                                                     | 41     | 1200 | 1000  | 19 ms | 24 ms    |
| XGBoost Higgs                                                                            | 28     | 500  | 1000  | ?     | 14 ms    |

（`?`）XGBoost 的多线程预测目前无法通过 Python 绑定启用。（`*`）KDD Cup 99 同时含连续与类别特征。

Born 后端的微基准（Windows / WebGPU）见 [docs/benchmark-baseline.md](docs/benchmark-baseline.md)。

## 局限

- **LightGBM** —— 支持的变换函数有限（仅 sigmoid 与 softmax）
- **XGBoost** —— 支持的变换函数有限（仅 sigmoid 与 softmax）；浮点转换的精度差异会导致 C API 与 `_leaves_` 的预测有零星 ULP 级偏差
- **scikit-learn** —— 不支持变换函数（输出为 `GradientBoostingClassifier.decision_function` 的原始分数）；仅支持 pickle 协议 `0`；同样有浮点精度差异

## 项目演进

路线图现已转为“当前代码库审计 + 下一年演进计划”的写法。见 [演进计划.md](演进计划.md) 的 **v5.4**，以及作为历史已完成清单的 [TODO.md](TODO.md)。

## 联系方式

有任何问题或想交流，可发邮件：**steper@foxmail.com**。

## 许可证

[MIT](LICENSE.md) —— Copyright (c) 2018 Steper Lin。
