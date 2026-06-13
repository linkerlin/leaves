# leaves

[![版本](https://img.shields.io/badge/版本-0.8.0-yellow.svg)](https://semver.org)
[![构建状态](https://travis-ci.org/dmitryikh/leaves.svg?branch=master)](https://travis-ci.org/dmitryikh/leaves)
[![Go 文档](https://godoc.org/github.com/dmitryikh/leaves?status.png)](https://godoc.org/github.com/dmitryikh/leaves)
[![测试覆盖率](https://coveralls.io/repos/github/dmitryikh/leaves/badge.svg?branch=master)](https://coveralls.io/github/dmitryikh/leaves?branch=master)
[![Go 代码质量](https://goreportcard.com/badge/github.com/dmitryikh/leaves)](https://goreportcard.com/report/github.com/dmitryikh/leaves)

![Logo](logo.png)

## 引言

_leaves_ 者，**纯 Go** 所写之 GBRT（梯度提升回归树）模型预测库也。其志在使人不假 C 语言 API 绑定，亦能于 Go 程序中调用时下流行 GBRT 框架之模型。

**注意**：`1.0.0` 版本发布以前，API 或有变动。

## 功能

  * 通用功能：
    * 支持批量并行预测
    * 支持 sigmoid 与 softmax 变换函数
    * 支持提取决策树之叶节点索引
  * 支持 LightGBM（[代码仓库](https://github.com/Microsoft/LightGBM)）模型：
    * 可从 `text` 格式及 `JSON` 格式读入模型
    * 支持 `gbdt`、`rf`（随机森林）与 `dart` 模型
    * 支持多分类预测
    * 针对类别特征有额外优化（如 _独热_ 决策规则）
    * 针对纯预测场景有额外优化
  * 支持 XGBoost（[代码仓库](https://github.com/dmlc/xgboost)）模型：
    * 可从二进制格式读入模型
    * 支持 `gbtree`、`gblinear`、`dart` 模型
    * 支持多分类预测
    * 支持缺失值（`nan`）
  * 支持 scikit-learn（[代码仓库](https://github.com/scikit-learn/scikit-learn)）树模型（实验性支持）：
    * 可从 pickle 格式（协议 `0`）读入模型
    * 支持 `sklearn.ensemble.GradientBoostingClassifier`

## 用法示例

起步之先，取此仓库于本地：

```sh
go get github.com/dmitryikh/leaves
```

极简示例：

```go
package main

import (
	"fmt"

	"github.com/dmitryikh/leaves"
)

func main() {
	// 1. 读入模型
	useTransformation := true
	model, err := leaves.LGEnsembleFromFile("lightgbm_model.txt", useTransformation)
	if err != nil {
		panic(err)
	}

	// 2. 执行预测！
	fvals := []float64{1.0, 2.0, 3.0}
	p := model.PredictSingle(fvals, 0)
	fmt.Printf("预测结果 %v: %f\n", fvals, p)
}
```

若用 XGBoost 模型，仅需将 `leaves.LGEnsembleFromFile` 换为 `leaves.XGEnsembleFromFile` 即可。

### XGBoost 3.x 模型加载（JSON / UBJ）

`io.LoadFromFile` 自动识别格式（亦可用 `io.DetectFormat`）：

| 格式 | 扩展名 | 说明 |
|------|--------|------|
| XGBoost JSON | `.json` | 3.x `save_model` 默认；`base_score` 在 logistic 目标下自动转 margin |
| XGBoost UBJSON | `.ubj` | 与 JSON 等价二进制；推理结果与 JSON 一致 |
| XGBoost 二进制 | 无扩展 / `.bin` | 经典 `xgb.Booster.save_model` 旧格式 |
| LightGBM | `.txt` / `.model` / `.json` | text 与 JSON 均支持 |

```go
import "github.com/dmitryikh/leaves/io"

// JSON 或 UBJ：同一套 API
m, err := io.LoadFromFile("model.ubj", &io.LoadOptions{
    LoadTransformation: true,  // binary:logistic → sigmoid 后概率
    Backend:            io.BackendAuto, // 小模型 Native，大 batch 可 BornCPU/GPU
})
```

`LoadTransformation: false` 时 `OutputValue` 与 `OutputMargin` 均为 raw margin；contrib 始终在 **margin 空间**。

### 训练能力边界（v2.x）

| 阶段 | 范围 | 状态 |
|------|------|------|
| **T1** | hist/exact + gbtree + 基础目标函数 | ✅ |
| **T2** | dart/gblinear + 多目标扩展 | ✅ |
| **T3** | CV/早停/checkpoint + XGB 3.x JSON 导出 | ✅ |
| **T4** | Born hist 增益（`born_train` build tag） | ✅ |
| **T5** | rank/survival、外存 DMatrix、单调约束 | rank 训练 ✅；survival/外存/单调约束 未开始 |

训练入口：`train.NewLearner` → `Learner.Fit` → `Learner.Save` / `io.ExportXGBoostJSON`。验收：`go test ./train/...` 与 `go test -tags born_train ./treebuilder/...`。

### 计算底座

推理与训练加速基于 [Born](https://github.com/born-ml/born)（CPU SIMD + WebGPU）。`NativeEngine` 为正确性基准；`BackendAuto` 在小 batch / 小模型下默认 **Native**，大 batch（≥256）且纯数值树时可选用 `BackendBornGPU`（Windows）。

**parity 门禁**：`TestBornParityFormatMatrix` 覆盖 LGB text/JSON、XGB bin/json/ubj、SK pickle × batch 1/16/256 × BornCPU/GPU，容差 `1e-5`（相对 Native）。

```go
m, _ := leaves.LoadFromFile("model.json", &io.LoadOptions{
    Backend: io.BackendBornCPU,
})
```

**Born 后端实测（`lg_breast_cancer`，i9-13900HX，Windows）** — `go test -bench=BenchmarkBreastCancerBackend -run=^$`：

| batch | Native | BornCPU | BornGPU |
|-------|--------|---------|---------|
| 1 | ~3 µs/op | ~1.9 ms/op | ~1.1 s/op（含 WebGPU 初始化） |
| 16 | ~82 µs/op | ~4.3 ms/op | ~1.7 s/op |

小森林张量路径有固定开销；**生产选型建议**：默认 `BackendAuto` 或 `BackendNative`；大 batch 数值树再测 BornGPU。训练加速：`go test -tags born_train ./treebuilder/...`

### Tree SHAP 与可解释性（v1.1+）

推荐经 `io.LoadFromFile` 加载后使用 `model.Ensemble.Explain()`：

```go
import (
	"github.com/dmitryikh/leaves"
	"github.com/dmitryikh/leaves/explain"
)

m, _ := leaves.LoadFromFile("model.json", leaves.DefaultLoadOptions())

x := [][]float64{{1.0, 2.0, 3.0}}
contrib, _ := m.Explain().TreeSHAP(x)       // margin 空间 SHAP
inter, _ := m.Explain().InteractionSHAP(x)  // 交互值矩阵
mc, _ := m.Explain().TreeSHAPMulticlass(x)  // 多类：[sample][feature][class]
base := m.Explain().ExpectedValue()         // 背景基线（全零特征）

imp := m.Explain().Importance(explain.ImportanceGain, nil)
text := m.Explain().DumpText(nil)
dot := m.Explain().DumpDOT(nil)
```

说明：SHAP 采用 **Lundberg 快速 Tree SHAP**（`SumHess` 覆盖权重，tree_path_dependent）；背景基线仍为全零特征。与 XGBoost `pred_contribs` 在数值上可能略有差异，但可加性 `base + Σφ ≈ margin` 成立。

**推荐（`predict.Request` 统一出口）**：末列 `bias` 为背景 margin（全零特征）；与 XGBoost `pred_contribs` **可加性一致**，逐元素分解可能不同。

```go
import "github.com/dmitryikh/leaves/predict"

nf := m.NFeatures()
nRows := 1

// 概率 / 变换后输出
vals := make([]float64, nRows*m.NOutputGroups())
_ = m.PredictWithRequest(predict.Request{
    Matrix: predict.DenseMatrix{Values: flat, Rows: nRows, Cols: nf},
    Output: predict.OutputValue,
}, vals)

// raw margin（未 sigmoid/softmax）
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

// Tree SHAP 贡献值（margin 空间；二分类 [sample][feature+bias]）
out := make([]float64, (nf+1)*nRows)
_ = m.PredictWithRequest(predict.Request{
    Matrix: predict.DenseMatrix{Values: flat, Rows: nRows, Cols: nf},
    Output: predict.OutputContribution, // 或 OutputApproxContribution / OutputInteraction
}, out)

// 多分类：扁平布局 [sample][class][feature+bias]，class 维 = NRawOutputGroups()
ng := m.NRawOutputGroups()
outMC := make([]float64, nRows*ng*(nf+1))
```

**bias 语义**：leaves 末列 = 全零特征的 **背景 margin**（`Explain().ExpectedValue()`）；与 XGBoost `pred_contribs` 可加性一致，逐元素分解可能不同。

### 评估指标 `metrics/`

内置 RMSE/MAE/AUC/LogLoss/MAPE/RMSLE/NDCG@k/MAP 等，名称与 XGBoost `eval_metric` 对齐（`metrics.Resolve`）：

```go
import "github.com/dmitryikh/leaves/metrics"

rmse, _ := metrics.RMSE{}.Evaluate(yTrue, yPred)
m, _ := metrics.Resolve("ndcg@5", metrics.Options{Groups: []int{10, 10}})
ndcg, _ := m.Evaluate(yTrue, yPred)
```

训练 `train.Config.EvalMetric` 支持同上名称；排序指标需 `data.GroupedMatrix` 提供 `Groups()`。

### 排序学习（`rank:pairwise` / `rank:ndcg`）

对标 XGBoost LambdaMART：组内 pairwise logistic + 可选 `|ΔNDCG|` 缩放。数据须为 `data.GroupedMatrix`（`Groups()` 为每个 query 的文档数）。

> **listwise 说明**：XGBoost / leaves **没有** `rank:listwise` 目标名。文献里的 listwise（按整组 query 优化 NDCG/MAP）请用 **`rank:ndcg`**（LambdaMART）；`rank:pairwise` 是 RankNet 纯 pairwise 损失。

```go
import (
    "github.com/dmitryikh/leaves/data"
    "github.com/dmitryikh/leaves/train"
)

// TSV：qid label feat1 feat2 ...（同 qid 行连续）
dm, _ := data.LoadRankingTSV("rank_train.tsv", "\t")

learner, _ := train.NewLearner(train.Config{
    Objective:    train.ObjectiveRankNDCG, // listwise 对标；或 rank:pairwise
    NumRound:     100,
    MaxDepth:     6,
    LearningRate: 0.1,
    NDCGK:        10,              // eval + lambda ndcg@10
    LambdaRankNorm: true,          // lambdarank_norm（默认开启）
    TreeMethod:   train.TreeMethodHist,
    // Subsample 在排序目标下自动置 1.0（query 须完整）
})
_ = learner.Fit(dm)
_ = learner.Save("rank.leaves.json")
```

#### MovieLens 100K Demo（电影评分 → 个性化排序）

每个 **user = query**，每条评分 = 文档；**label = 1–5 星**（多档相关性，适合 `rank:ndcg` listwise）。

```bash
cd testdata && python gen_rank_movielens.py   # 下载 ml-100k.zip 到 .cache/
go test ./train/... -run 'TestRankMovieLens' -v
```

特征：电影流行度、均分、年份、19 维类型 one-hot（22 维）。

早停：`EvalSet` 也须带 `Groups()`；`EarlyStop.Maximize` 会随 NDCG 等指标自动设为 `true`。

**验收**：`testdata/gen_rank_smoke.py` 合成数据；`gen_rank_msltr.py` MSLR 子集；`gen_rank_movielens.py` MovieLens 100K listwise demo。

```bash
cd testdata && python gen_rank_smoke.py
cd testdata && python gen_rank_movielens.py  # MovieLens listwise demo（~5MB）
cd testdata && python gen_rank_msltr.py      # 首次 ~1.2G zip
go test ./train/... -run 'Rank.*' -v
go test ./train/... -short                   # 跳过 MSLTR 慢测
```

## 文档

文档托管于 godoc（[链接](https://godoc.org/github.com/dmitryikh/leaves)）。其中列有复杂用法示例及完整 API 参考。另于 [leaves_test.go](leaves_test.go) 中可寻得若干用法信息。

## 兼容性

_leaves_ 诸多功能均已测过，能与各版 GBRT 库新旧兼容。[compatibility.md](compatibility.md) 中详载了 _leaves_ 对不同版本外部 GBRT 库之正确性校验报告。

关于新功能与向后兼容性的补充说明，见 [NOTES.md](NOTES.md)。

## 性能比对

下列乃批量预测速度对照（单次 API 调用约含 1000 条数据）。硬件环境：MacBook Pro（15 英寸，2017 款），2.9 GHz Intel Core i7，16 GB 2133 MHz LPDDR3。C 语言 API 实现经由 Python 绑定调用；然大批量情形下 Python 绑定之开销可略去不计。_leaves_ 的性能测试以 Go 语言测试框架驱动：`go test -bench`。详情见 [benchmark](benchmark) 目录，数据准备流程见 [testdata/README.md](testdata/README.md)。

单线程：

| 测试用例 | 特征数 | 树数 | 批量大小 | C API | _leaves_ |
|----------|--------|------|----------|-------|----------|
| LightGBM [MS LTR](https://github.com/Microsoft/LightGBM/blob/master/docs/Experiments.rst#comparison-experiment) | 137 | 500 | 1000 | 49ms | 51ms |
| LightGBM [Higgs](https://github.com/Microsoft/LightGBM/blob/master/docs/Experiments.rst#comparison-experiment) | 28 | 500 | 1000 | 50ms | 50ms |
| LightGBM KDD Cup 99* | 41 | 1200 | 1000 | 70ms | 85ms |
| XGBoost Higgs | 28 | 500 | 1000 | 44ms | 50ms |

四线程：

| 测试用例 | 特征数 | 树数 | 批量大小 | C API | _leaves_ |
|----------|--------|------|----------|-------|----------|
| LightGBM [MS LTR](https://github.com/Microsoft/LightGBM/blob/master/docs/Experiments.rst#comparison-experiment) | 137 | 500 | 1000 | 14ms | 14ms |
| LightGBM [Higgs](https://github.com/Microsoft/LightGBM/blob/master/docs/Experiments.rst#comparison-experiment) | 28 | 500 | 1000 | 14ms | 14ms |
| LightGBM KDD Cup 99* | 41 | 1200 | 1000 | 19ms | 24ms |
| XGBoost Higgs | 28 | 500 | 1000 | ? | 14ms |

（?）—— 目前尚无法通过 Python 绑定利用 XGBoost 之多线程预测。

（*）—— KDD Cup 99 问题同时涉及连续特征与类别特征。

## 局限

  * LightGBM 模型：
    * 变换函数支持有限（仅 sigmoid 与 softmax）
  * XGBoost 模型：
    * 变换函数支持有限（仅 sigmoid 与 softmax）
    * 受浮点数转换与比较精度之差，C API 预测结果与 _leaves_ 或有微殊
  * scikit-learn 树模型：
    * 不支持变换函数，输出分数为 _原始分数_（即 `GradientBoostingClassifier.decision_function` 所出者）
    * 仅支持 pickle 协议 `0`
    * 受浮点数转换与比较精度之差，sklearn 预测结果与 _leaves_ 或有微殊

## 项目演进

全链路路线图（对标 XGBoost 3.3：训练·推理·线上·观测）见 [`演进计划.md`](演进计划.md)；可执行 backlog 见 [`TODO.md`](TODO.md)。

## 联系方式

若有兴趣于此项目，或有疑问，可发邮件致：steper@foxmail.com
