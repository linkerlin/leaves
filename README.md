# leaves

[![版本](https://img.shields.io/badge/版本-v2.x--dev-blue.svg)](https://semver.org)
[![构建状态](https://github.com/dmitryikh/leaves/actions/workflows/ci.yml/badge.svg)](https://github.com/dmitryikh/leaves/actions/workflows/ci.yml)
[![Go 文档](https://godoc.org/github.com/dmitryikh/leaves?status.png)](https://godoc.org/github.com/dmitryikh/leaves)
[![测试覆盖率](https://coveralls.io/repos/github/dmitryikh/leaves/badge.svg?branch=master)](https://coveralls.io/github/dmitryikh/leaves?branch=master)
[![Go 代码质量](https://goreportcard.com/badge/github.com/dmitryikh/leaves)](https://goreportcard.com/report/github.com/dmitryikh/leaves)

![Logo](logo.png)

## 引言

_leaves_ 者，**纯 Go** 所写之 GBRT（梯度提升回归树）**训练与推理**库也。支持加载 XGBoost / LightGBM / scikit-learn 模型，亦可自训练（hist/exact、rank、survival、tweedie 等），无需 C 语言 API 绑定。

**推荐入口**：`leaves.LoadFromFile`（推理，默认按 objective 自动变换）、`leaves.LoadDataAuto` / `leaves.NewLearner`（训练）。遗留 API `LGEnsembleFromFile` / `XGEnsembleFromFile` 仍可用。

## 功能

  * 通用功能：
    * 支持批量并行预测
    * 支持 sigmoid 与 softmax 变换函数；`DefaultLoadOptions()` 默认 **AutoTransform**（按 objective 自动启用）
    * 支持提取决策树之叶节点索引
    * **训练数据内容嗅探**（libsvm / 排序 TSV / CSV·TSV，扩展名仅作回退）
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
  * **训练**（v2.x，T1–T5）：
    * `gbtree` / `dart` / `gblinear`；hist / exact / gpu_hist
    * 目标：regression、binary/multi、gamma/poisson、tweedie、survival:cox/aft、rank
    * CV、早停、checkpoint 续训、外存 DMatrix、单调约束、max_leaves
    * 导出 `leaves.json` / XGBoost 3.x JSON
  * **部署与观测**：
    * int8 量化（parity 门禁）、`Ensemble.Reload`、推理 profiling
    * WASM demo（`examples/wasm/`）、HTTP embed（`examples/http/`）、训练便利 demo（`examples/train_from_model/`）
  * 支持 scikit-learn（[代码仓库](https://github.com/scikit-learn/scikit-learn)）树模型（实验性）：
    * 可从 pickle 格式（协议 `0`）读入模型
    * 支持 `sklearn.ensemble.GradientBoostingClassifier`

## 用法示例

起步之先，取此仓库于本地：

```sh
go install github.com/linkerlin/leaves@latest
```

极简示例（推荐 `LoadFromFile`）：

```go
package main

import (
	"fmt"

	"github.com/dmitryikh/leaves"
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
	fmt.Printf("预测结果 %v: %f\n", fvals, p)
}
```

遗留 API 仍可用：`leaves.LGEnsembleFromFile(path, loadTransformation)` / `leaves.XGEnsembleFromFile(...)`。

根包训练别名（与 `train` 包等价）：

```go
dm, _ := leaves.LoadDataAuto("train.tsv") // 内容嗅探

learner, _ := leaves.NewLearner(leaves.TrainConfig{
	Objective: leaves.TrainObjectiveBinaryLogistic,
	NumRound:  50,
})
_ = learner.Fit(dm)
```

### 格式自动识别（v4.3）

**推理**：`io.LoadFromFile` / `io.DetectFormat` 按扩展名与文件头识别模型；经典 XGB 二进制无魔数时用 header 探测；`.pkl`/`.joblib` → sklearn；数值表误作 `.txt` 模型时明确报错。

**训练**：`data.FromFileAuto` / `leaves.LoadDataAuto`，`FormatAuto` 时读文件样本推断格式（见下表）。

| 嗅探结果 | 典型内容 | 加载函数 |
|----------|----------|----------|
| LIBSVM | `label 1:0.5 3:1.2` | `FromLIBSVM` |
| 排序 TSV | `qid label feat...` | `LoadRankingTSV` |
| TSV 末列 label | 无表头、末列为 label | `LoadDenseTSV` |
| CSV | 有表头或启发式 label 列 | `FromCSV` |

**变换**：`DefaultLoadOptions()` 已设 `AutoTransform: true`——`binary:logistic` 等返回概率，`reg:squarederror` 仍 raw。要 margin 请设 `AutoTransform: false` 或用 `predict.OutputMargin`。

### XGBoost 3.x 模型加载（JSON / UBJ）

`io.LoadFromFile` 自动识别格式（亦可用 `io.DetectFormat`）：

| 格式 | 扩展名 | 说明 |
|------|--------|------|
| XGBoost JSON | `.json` | 3.x `save_model` 默认；`base_score` 在 logistic 目标下自动转 margin |
| XGBoost UBJSON | `.ubj` | 与 JSON 等价二进制；推理结果与 JSON 一致 |
| XGBoost 二进制 | 无扩展 / `.bin` / `.model` | 经典 `xgb.Booster.save_model`；无魔数时按 header 探测 |
| LightGBM | `.txt` / `.model` / `.json` | text 与 JSON 均支持 |
| sklearn | `.pkl` / `.joblib` | pickle 魔数识别 |

```go
import "github.com/dmitryikh/leaves/io"

// JSON 或 UBJ：DefaultLoadOptions 已默认 AutoTransform
m, err := io.LoadFromFile("model.ubj", io.DefaultLoadOptions())

// 显式控制变换
m, err = io.LoadFromFile("model.ubj", &io.LoadOptions{
    AutoTransform:      false, // 输出 raw margin
    LoadTransformation: true,  // 或强制对所有目标启用变换
    Backend:            io.BackendAuto,
})
```

`AutoTransform: false` 且 `LoadTransformation: false` 时 `OutputValue` 与 `OutputMargin` 均为 raw margin；contrib / SHAP 始终在 **margin 空间**（与是否 AutoTransform 无关）。

**3.1+ 互操作要点**（2026-06-15）：

| 能力 | 状态 |
|------|------|
| gbtree / dart JSON·UBJ | ✅（含 3.x `gbtree.model` 嵌套 DART） |
| gblinear JSON·UBJ | ✅ |
| 经典二进制 gbtree/dart/gblinear | ✅ 统一走 `ForestIR`/`LinearIR` |
| 原生 categorical + `model.cats` | ✅ 加载；导出保留原文 round-trip |
| multi:softprob / multi:softmax | ✅ 概率向量 / argmax 类标签 |
| multi-output 向量叶 | ✅ |
| `reg:gamma` / `count:poisson` / `reg:tweedie` 加载变换 | ✅ Exp + `base_score` log 转换 |
| `AutoTransform` 默认 + sklearn `.pkl` 识别 | ✅ v4.3 |
| 导出 `loss_changes` / `sum_hessian` | ✅（自训练树） |

### 训练能力边界（v2.x）

| 阶段 | 范围 | 状态 |
|------|------|------|
| **T1** | hist/exact + gbtree + 基础目标函数 | ✅ |
| **T2** | dart/gblinear + 多目标扩展 | ✅ |
| **T3** | CV/早停/checkpoint + XGB 3.x JSON 导出 | ✅ |
| **T4** | 全局分箱 + Born/WebGPU hist + GPU margin 预测 | ✅ |
| **T5** | rank 训练 + 单调约束 + survival / 外存 / tweedie + 续训 API | ✅ |

训练入口：`train.NewLearner` → `Learner.Fit` → `Learner.Save` / `io.ExportXGBoostJSON`。  
数据加载：`data.FromFileAuto` / `leaves.LoadDataAuto`（**内容嗅探** CSV·TSV·libsvm·排序 TSV）；亦可显式 `data.FromCSV` / `data.FromLIBSVM`。  
续训：`train.LoadCheckpoint` / `train.ResumeFit`；评估：`learner.Eval(dm)`。  
`MaxLeaves` 启用 lossguide 单树叶子预算。默认 `TreeMethod=auto`（小数据 exact，≥5 万行 hist）。验收：`go test ./train/...`。

**外存 DMatrix**（多批次 hist，labels 在内存、特征按批 `RowAt`）：

```go
bm, _ := data.SplitDense(dense, batchRows) // 或 data.NewBatchedMatrix(batches, labels, weights)
learner, _ := train.FitExternal(train.Config{
    Objective:     train.ObjectiveBinaryLogistic,
    TreeMethod:    train.TreeMethodHist,
    HistBinPolicy: "global",
    NumRound:      10,
}, bm)
```

**Checkpoint 续训**：

```go
learner, _ := train.ResumeFit("ckpt.json", train.Config{
    Objective: train.ObjectiveSquaredError,
    NumRound:  20, // 在 checkpoint 已完成轮次上继续
    TreeMethod: train.TreeMethodExact,
}, dm)
score, _ := learner.Eval(dm)
```

**数据文件**（自动嗅探，扩展名仅作回退）：

```go
// 推荐：一行加载，自动识别 libsvm / 排序 TSV / 末列 label 的 TSV / CSV
dm, _ := leaves.LoadDataAuto("train.tsv")

// 显式选项（FormatAuto 时仍会做内容嗅探）
dm, _ := data.FromFile("train.csv", data.FileLoadOptions{
    Format: data.FormatAuto,
    CSV:    data.CSVOptions{HasHeader: true, HasLabelColumn: true, LabelCol: 0},
})
csr, _ := data.FromLIBSVM("train.libsvm", data.LIBSVMOptions{HasLabel: true})
```

嗅探规则摘要：`index:value` 稀疏行 → libsvm；`qid label feat...` → 排序；无表头 TSV 末列 label → `LoadDenseTSV`；其余 → CSV（分隔符与 label 列启发式）。

**从模型推断 objective 并训练**（[`examples/train_from_model`](examples/train_from_model/README.md)）：

```go
learner, _ := leaves.NewLearnerFromModelAndData(
    "reference_model.json",  // 仅读取 objective（XGB JSON / UBJ / leaves.json）
    "train.tsv",             // LoadDataAuto 嗅探
    leaves.TrainConfig{NumRound: 10, TreeMethod: leaves.TrainTreeMethodExact},
    data.DefaultFileLoadOptions(),
)
_ = learner.Save("out.leaves.json")
```

```bash
go run ./examples/train_from_model/ -model testdata/xgboost_smoke.json -data testdata/breast_cancer_train.tsv
```

**survival:aft 区间删失**（`data.AFTDense`）：

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

#### 训练加速（T4+）

hist / `gpu_hist` 路径上的加速栈（日志前缀 `[leaves/train] accel:`）：

| 阶段 | 实现 | 默认 `auto`（≥3 万行 + WebGPU） | `LEAVES_TRAIN_ACCEL=webgpu` |
|------|------|-------------------------------|------------------------------|
| 全局分箱 | 训练期每特征一次切点 + 行级 bin 缓存 | ✅ | ✅ |
| 直方图累加 | CPU rowBin / WebGPU SelectAdd 批量 | **自动 gpu_hist** | GPU batch |
| 增益扫描 | 纯 CPU / 批量 WebGPU gain | **批量 GPU**（GPU 路径） | 批量 GPU |
| margin 预测 | 逐行 CPU / Born GPU `PredictDense` | `PredictMargins` ≥256 行 | ≥64 行 |

`AccelMode=auto` 且 `TreeMethod=auto` 时：行数 **≥30000** 且 WebGPU 可用 → 自动解析为 `gpu_hist` + `webgpu`；小数据或显式 `TreeMethod=hist` 保持 CPU 路径。

```go
learner, _ := train.NewLearner(train.Config{
    Objective:    train.ObjectiveRankNDCG,
    TreeMethod:   train.TreeMethodGPUHist,
    AccelMode:    train.AccelModeWebGPU, // 或依赖环境变量 LEAVES_TRAIN_ACCEL
    NumThreads:   4,                     // 并行分裂评估 + GPU batch hist
    HistBinPolicy: "global",              // 默认；per_node 恢复旧 per-node 切点
    NumRound:     50,
})
_ = learner.Fit(dm)
```

环境变量：`LEAVES_TRAIN_ACCEL=auto|webgpu|born_cpu|cpu`（覆盖 `Config.AccelMode`）。

**单调约束**（T5，`MonotoneConstraints []int`，对标 XGBoost `monotone_constraints`）：

```go
learner, _ := train.NewLearner(train.Config{
    Objective:           "reg:squarederror",
    MonotoneConstraints: []int{1, 0, -1}, // 特征 0 递增，2 递减
    TreeMethod:          train.TreeMethodHist,
})
```

**MSLTR 子集加速对比**（120 train queries / 5425 docs，50 轮 `rank:ndcg`，`i9` 类 Windows + WebGPU；复现：`go test ./train/... -run TestMSLTRTrainAccelBenchmark -v`）：

| 模式 | 耗时 | train NDCG@10 | test NDCG@10 |
|------|------|---------------|--------------|
| `cpu_hist` | **128s** | 0.706 | 0.397 |
| `auto_hist` | ~142s | 0.697 | 0.397 |
| `webgpu_hist` | 362s | 0.694 | 0.387 |

实测环境：Windows + WebGPU，`rank:ndcg` 50 轮，5425 文档 / 136 特征，`NumThreads=4`（2026-06-13 本机复现）。

- **`cpu_hist` 最快**：纯 CPU 增益扫描 + 全局 rowBin，无 Born 张量 / GPU 往返开销。
- **`auto`**：与 `cpu_hist` 耗时接近（boosting 内不走 GPU margin）；GPU margin 仅在 `PredictMargins` / `EvalSet` 启用。
- **`webgpu_hist`**：见下文「为何中小数据集上 GPU 未更快」。

**为何 MSLTR 子集上 `webgpu_hist` 比 CPU 慢？**

一部分是**数据规模偏小**，但不只是行数少：

| 因素 | MSLTR 子集 | GPU 更划算的典型场景 |
|------|------------|----------------------|
| 训练行数 | ~5k | 10 万+ |
| 每节点样本 | 递归分裂后大量节点 < 64 | 根/浅层节点常有数千～上万样本 |
| 特征数 | 136 | 数百列时 batch 收益更大 |

更关键的是 **GPU 实际只覆盖了一小部分工作**。实测 `webgpu_hist` 一轮 Fit 日志示例：

```
hist_build(cpu=341320 webgpu=57024 total=398344)  → 仅 ~14% 直方图走 GPU
gain_scan(webgpu=0 born_cpu=0 pure_cpu=324088)    → 增益扫描 0% GPU（NumThreads=4）
```

原因包括：

1. **节点子集 < 64 样本** → 强制 CPU 直方图（浅层 `gpuHistMinSamples=64`，深层略降）；树越深，越多节点落在 CPU 路径。
2. **中等节点混合策略**：样本数 < 4096 的节点仅前 32 特征走 GPU batch，其余 CPU rowBin（避免小节点 GPU 固定开销反超）。
3. **`NumThreads>1`** → WebGPU 增益扫描仍禁用（小直方图上 Born/GPU 固定开销 > 纯 CPU 循环）；`webgpu_t1` 可走 GPU 增益扫描。
4. **异构队列 + 批量 GPU gain scan**：hist batch 后同 mutex 内将 chunk（≤64 特征）的 f32 直方图 **一次 2D 上传**算增益，跳过逐特征 GPU 会话；`hasGain` 直出分裂点。
5. **`NumThreads>1`**：GPU 路径增益走批量 scan（~53%）；CPU 路径特征仍纯 CPU 增益扫描。

因此在该子集上出现 **362s vs 128s** 并不意外——不是 GPU 无效，而是**工作集太小 + GPU 命中率低 + 并行 hist 不走 GPU 增益扫描**。

**何时 `webgpu_hist` 更可能反超 `cpu_hist`**

需同时接近以下条件：

- 训练行数 **10 万+**（或浅层节点样本经常 ≥ 几千）
- 大 `EvalSet` / `PredictMargins` 走 GPU margin（≥64/256 行阈值）
- 更多节点命中 GPU hist（可调大 `MaxBin`、或后续降低 `gpuHistMinSamples`）
- 显式 `LEAVES_TRAIN_ACCEL=webgpu` + `TreeMethod=gpu_hist`；若需 GPU 增益扫描，暂用 `NumThreads=1`（与多线程确定性/trade-off 相关）

**选型建议**：MSLTR 量级及同规模表格数据，优先 `cpu_hist` 或 `auto`；十万行以上再测 `webgpu_hist`。复现对比：

```bash
# MSLTR 子集（~5k 行）；默认 go test ./... 会跳过，需显式开启
$env:LEAVES_BENCH=1
go test ./train/... -run TestMSLTRTrainAccelBenchmark -v -timeout 45m

# 大规模稠密回归交叉点（默认 5 万行 × 64 特征 × 10 轮）
go test ./train/... -run TestLargeDenseTrainAccelBenchmark -v -timeout 60m
# 可调规模：LEAVES_BENCH_ROWS / LEAVES_BENCH_COLS / LEAVES_BENCH_ROUNDS
# 单模式：LEAVES_BENCH_ONLY=cpu_hist|webgpu_hist|webgpu_t1
```

大规模 benchmark 会输出 `hist_gpu=%`（直方图走 GPU 的比例）与完整 `accel summary`，便于判断交叉点。

**大规模稠密回归对比**（`reg:squarederror`，64 特征，`MaxDepth=6`，Windows WebGPU；`TestLargeDenseTrainAccelBenchmark`）：

| 行数 | 轮数 | `cpu_hist` | `webgpu_hist` | hist GPU 占比 | 结论 |
|------|------|------------|---------------|---------------|------|
| 2 万 | 5 | **4.3s** | 8.0s | ~49% | CPU 更快（固定开销主导） |
| 5 万 | 10 | **15.3s** | 28.1s | ~61% / gain ~53% | GPU 路径批量 gain；总耗时仍略慢于 CPU |
| 5 万 | 10 | — | **29.0s** (`auto_smart`) | ~61% / gain ~53% | `AccelMode=auto` + `TreeMethod=auto` 自动解析为 `gpu_hist`+`webgpu` |
| 10 万 | 10 | 63.2s | **59.2s** | ~49% | GPU 约快 6% |

交叉点约在 **3–5 万行**（本机合成数据、浅层节点仍约一半走 CPU hist）。行数继续增大时收益受混合策略与 `NumThreads=4`（增益扫描走纯 CPU）限制；`LEAVES_BENCH_ONLY=webgpu_t1` 可启用 GPU 增益扫描，但单线程在大数据上通常更慢。

### 计算底座

推理与训练加速基于 [Born](https://github.com/born-ml/born)（CPU SIMD + WebGPU）。**训练**：见上文 [训练加速（T4+）](#训练加速t4)；`Fit` 结束日志含 `gain_scan` / `hist_build` / `accel margin` 分项。**推理**：`NativeEngine` 为 golden；`BackendAuto` 大 batch 可选用 `BackendBornGPU`（Windows）。

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

小森林张量路径有固定开销；**生产选型建议**：默认 `BackendAuto`（`DefaultLoadOptions()` 已启用 `AutoTransform`）；小 batch / 低延迟走 Native，大 batch 数值树再测 BornGPU。CI 门禁 `TestBenchGateBornCPUSlowerBatch1` 保证 batch=1 时 BornCPU ≥20× 慢于 Native。训练加速 benchmark：`LEAVES_BENCH=1 go test ./train/... -run TestMSLTRTrainAccelBenchmark -v`

#### 后端选型速查

| 场景 | 推荐 Backend | 说明 |
|------|--------------|------|
| 在线单条 / batch≤8 | `BackendAuto` 或 `BackendNative` | 延迟优先 |
| 批推理 ≥256 行、纯数值树 | `BackendAuto` + `Workload.BatchSize` | Windows 可试 BornGPU |
| WASM / js | `BackendNative` | Born 在 js 回退 Native |
| 含 LGB cat-small | `BackendNative` | Born 不支持 |

```go
m, _ := leaves.LoadFromFile("model.json", &io.LoadOptions{
    Backend: io.BackendAuto,
    Workload: tree.WorkloadHint{BatchSize: 1024, HasGPU: true},
})
```

### Tree SHAP 与可解释性（v1.1+）

推荐经 `io.LoadFromFile` 加载后使用 `model.Ensemble.Explain()`。`DefaultLoadOptions()` 对 logistic 模型返回**概率**；SHAP/contrib 仍在 margin 空间，与 `OutputMargin` / `predict.OutputContribution` 一致。

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

### 排序学习（`rank:pairwise` / `rank:ndcg` / `rank:listwise`）

对标 XGBoost LambdaMART：`rank:ndcg` / `rank:pairwise`。数据须为 `data.GroupedMatrix`（`Groups()` 为每个 query 的文档数）。

| 目标 | 类型 | 说明 |
|------|------|------|
| `rank:ndcg` | LambdaMART（XGBoost 兼容） | pairwise + \|ΔNDCG\| 缩放，优化排序指标 |
| `rank:pairwise` | RankNet pairwise | 默认 **topk**（k=32）+ **lambdarank_normalization**（对标 XGBoost）；可选 `LambdaRankPairMethod=full\|mean` |
| `rank:listwise` | **ListNet softmax CE**（leaves 原生） | 组内 q∝exp(label)、p=softmax(pred)，纯 listwise 交叉熵 |

> `rank:ndcg` 与 `rank:listwise` 都面向 listwise 排序，但损失不同：前者对标 XGBoost LambdaMART，后者为 ListNet 式 softmax 目标（XGBoost 无同名目标）。

```go
import (
    "github.com/dmitryikh/leaves/data"
    "github.com/dmitryikh/leaves/train"
)

// TSV：qid label feat1 feat2 ...（同 qid 行连续）
dm, _ := data.LoadRankingTSV("rank_train.tsv", "\t")

// listwise（ListNet）
learner, _ := train.NewLearner(train.Config{
    Objective:    train.ObjectiveRankListwise,
    NumRound:     40,
    MaxDepth:     4,
    LearningRate: 0.1,
    NDCGK:        10,
    EvalMetric:   "ndcg@10",
})
_ = learner.Fit(dm)

// 或 RankNet pairwise（默认 topk 配对，k=32）
learner, _ = train.NewLearner(train.Config{
    Objective:    train.ObjectiveRankPairwise,
    NumRound:     40,
    MaxDepth:     4,
    LearningRate: 0.1,
    EvalMetric:   "ndcg@10",
    TreeMethod:   train.TreeMethodHist,
})
_ = learner.Fit(dm)

// 经典全配对（显式 full）
learner, _ = train.NewLearner(train.Config{
    Objective:            train.ObjectiveRankPairwise,
    LambdaRankPairMethod: train.LambdaRankPairFull,
    NumRound:             40,
    MaxDepth:             4,
    LearningRate:         0.1,
    EvalMetric:           "ndcg@10",
})

// 可选：显式关闭 λ 归一化（默认 topk 下已开启）
learner, _ = train.NewLearner(train.Config{
    Objective:                  train.ObjectiveRankPairwise,
    LambdaRankPairMethod:       train.LambdaRankPairTopK,
    LambdaRankNumPairPerSample: 32,
    // LambdaRankNormalization 默认 true；full 配对默认 false
    NumRound:                   40,
    MaxDepth:                   4,
    LearningRate:               0.1,
})
_ = learner.Fit(dm)

// 或 LambdaMART listwise（XGBoost 对标）
learner, _ = train.NewLearner(train.Config{
    Objective:    train.ObjectiveRankNDCG,
    NumRound:     100,
    MaxDepth:     6,
    LearningRate: 0.1,
    NDCGK:        10,
    LambdaRankNorm: true,
    TreeMethod:   train.TreeMethodHist,
})
_ = learner.Fit(dm)
```

#### MovieLens 100K Demo（电影评分 → 个性化排序）

每个 **user = query**，每条评分 = 文档；**label = 1–5 星**。可对比三种目标。

**可运行 Demo**（训练 → 保存模型 → Top-K 推荐）：见 [`demos/movielens/README.md`](demos/movielens/README.md)

```bash
cd testdata && python gen_rank_movielens.py
go run ./demos/movielens/cmd/train
go run ./demos/movielens/cmd/recommend -group 0 -topk 10
go test ./train/... -run 'TestRankMovieLens' -v
go test ./train/... -run TestFitRankingListwise -v
```

早停：`EvalSet` 也须带 `Groups()`；`EarlyStop.Maximize` 会随 NDCG 等指标自动设为 `true`。

**验收**：`testdata/gen_rank_smoke.py` 合成数据；`gen_rank_msltr.py` MSLR 子集；`gen_rank_movielens.py` MovieLens 100K listwise demo；`gen_rank_pairwise_grad.py` 逐对 λ golden；`gen_rank_ndcg_grad.py` rank:ndcg ΔNDCG golden。

```bash
cd testdata && python gen_rank_smoke.py
cd testdata && python gen_rank_pairwise_grad.py   # rank:pairwise 逐对 λ golden
cd testdata && python gen_rank_ndcg_grad.py       # rank:ndcg topk ΔNDCG golden
cd testdata && python gen_rank_movielens.py  # MovieLens listwise demo（~5MB）
cd testdata && python gen_rank_msltr.py      # 首次 ~1.2G zip
go test ./objective/... -run 'TestRank.*GradGolden' -v
go test ./train/... -run 'TestRankNDCGTopK|TestRankPairwiseTopK|Monotone|Callback' -v
go test ./train/... -run 'Rank.*' -v
go test ./train/... -short                   # 跳过 MSLTR rank trend 等慢测
# 训练加速 benchmark（~15–45min，默认 go test ./... 会 skip）：
#   $env:LEAVES_BENCH=1; go test ./train/... -run TestMSLTRTrainAccelBenchmark -v -timeout 45m
```

排序 + 单调约束：`MonotoneConstraints` 与 `rank:pairwise` / `rank:ndcg` / `rank:listwise` 可组合使用（见 `train/rank_monotone_test.go`）。

### 训练回调与学习率调度（P3）

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

内置调度器：`ExponentialLRScheduler(gamma)`、`StepLRScheduler(every, factor)`。  
`CallbackContext` 在配置 `EvalSet` + `EvalMetric` 时还会填充 `EvalMetric` / `EvalMetricOK`（无需 `EarlyStop`）。

### 推理 profiling 与模型热更新（P3）

```go
// 从 model.Ensemble 取得 NativeEngine 后 profiling
if ne, ok := m.Engine().(*tree.NativeEngine); ok {
    prof, err := tree.ProfileNativeDense(ne, vals, nrows, ncols, preds, 0)
    _ = prof.Elapsed
}

// 线上模型轮换（须 import github.com/dmitryikh/leaves 注册 loader）
_ = m.Reload("/path/to/new.model", io.DefaultLoadOptions())
```

`tree.ProfileWalkStats` 可单独统计单样本树遍历深度，不执行完整预测。

### int8 阈值量化与 parity 门禁

数值分裂阈值按特征做 int8 仿射量化（127 级），**叶子值保持 float64**；分类节点不量化。  
`quantize.Engine` 当前**不支持** `PredictLeafIndices*`（仅 margin 预测）。

```go
qf, _ := quantize.QuantizeForest(m.Forest(), quantize.Config{})
res, err := quantize.CheckParityWithGate(m.Forest(), qf, samples, 0, quantize.DefaultGate())

eng, _ := quantize.NewEngine(qf, nil, tree.TransformRaw, m.NOutputGroups())
model.NewEnsemble(eng) // 可替换线上 Ensemble 引擎
```

**P3 收尾验收**：

```bash
go test ./tree/... -run Profile -count=1
go test . -run TestEnsembleReload -count=1
go test ./quantize/... -count=1
go test ./train/... -run Callback -count=1
```


## 文档

| 文档 | 说明 |
|------|------|
| [godoc](https://godoc.org/github.com/dmitryikh/leaves) | API 参考 |
| [`演进计划.md`](演进计划.md) | 全链路路线图（**v4.3**，与代码同步） |
| [`TODO.md`](TODO.md) | 可执行 backlog（P0–T5 + v3.1 已清空） |
| [`NOTES.md`](NOTES.md) | 版本变更与兼容性说明（含 v4.3 AutoTransform） |
| [`compatibility.md`](compatibility.md) | 外部 GBRT 库正确性校验 |
| [`docs/testdata-matrix.md`](docs/testdata-matrix.md) | 回归测试矩阵 |
| [`docs/benchmark-baseline.md`](docs/benchmark-baseline.md) | Benchmark 与 CI 门禁 |
| [`examples/wasm/README.md`](examples/wasm/README.md) | WASM 部署指南 |
| [`examples/http/README.md`](examples/http/README.md) | HTTP embed 批预测 demo |
| [`examples/train_from_model/README.md`](examples/train_from_model/README.md) | 嗅探数据 + 推断 objective 训练 demo |
| [`leaves_test.go`](leaves_test.go) | 更多用法示例 |

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

全链路路线图（对标 XGBoost 3.3：训练·推理·线上·观测）见 [`演进计划.md`](演进计划.md)（**v4.3**：格式嗅探、`AutoTransform` 默认、训练便利 API）；可执行 backlog 见 [`TODO.md`](TODO.md)。

## 联系方式

若有兴趣于此项目，或有疑问，可发邮件致：steper@foxmail.com
