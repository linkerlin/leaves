# leaves

**English** | [中文](README.md)

[![Version](https://img.shields.io/badge/version-v2.5.3-blue.svg)](https://github.com/linkerlin/leaves/releases/tag/v2.5.3)
[![CI](https://github.com/linkerlin/leaves/actions/workflows/ci.yml/badge.svg)](https://github.com/linkerlin/leaves/actions/workflows/ci.yml)
[![Go Reference](https://pkg.go.dev/badge/github.com/linkerlin/leaves/v2.svg)](https://pkg.go.dev/github.com/linkerlin/leaves/v2)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE.md)

![Logo](logo.png)

## Introduction

`leaves` is a **pure Go** library for **training and inference of ranking
models for recommender systems**. It loads XGBoost, LightGBM, and scikit-learn
models out of the box, and ships a complete in-Go training loop (hist / exact,
ranking, survival, tweedie, …) — no C bindings required.

**Recommended entry points**

- Inference: `leaves.LoadFromFile` (auto-transform on by default), or the
  legacy `leaves.LGEnsembleFromFile` / `leaves.XGEnsembleFromFile`.
- Training: `leaves.NewLearner` / `train.NewLearner`, `leaves.LoadDataAuto` /
  `data.FromFileAuto` (content sniffing), and the convenience helper
  `leaves.NewLearnerFromModelAndData` that infers the objective from a
  reference model.

See [docs/testdata-matrix.md](docs/testdata-matrix.md) for the regression
matrix, [docs/api-surface.md](docs/api-surface.md) for recommended vs compat
APIs, [演进计划.md](演进计划.md) (v5.4) for the library roadmap,
[演进方案.md](演进方案.md) for the Agentic contract, and
[docs/release-checklist.md](docs/release-checklist.md) before tagging a release.

## Features

### General

- Parallel batched prediction (dense / CSR).
- Sigmoid and softmax transformations; `DefaultLoadOptions()` enables
  **AutoTransform** by default.
- Decision-tree leaf-index extraction.
- **Content sniffing** for training data (LIBSVM, ranking TSV, CSV·TSV — the
  extension is only a fallback).

### [LightGBM](https://github.com/microsoft/LightGBM)

- Loads both `text` and `JSON` formats.
- `gbdt`, `rf` (random forest), and `dart` model families.
- Multiclass predictions.
- Optimised decision rules for categorical features (e.g. *one-hot* splits).
- Specialised fast paths for prediction-only scenarios.

### [XGBoost](https://github.com/dmlc/xgboost)

- Loads binary, JSON, and [UBJSON](https://github.com/jmckaskill/ubjson) models.
- `gbtree`, `gblinear`, and `dart` boosters; nested DART `gbtree.model` (3.x).
- Multiclass predictions, multi-output vector leaves, XGBoost categorical
  (`model.cats`) with round-trip export.
- Missing-value (`nan`) support.
- Auto-loaded `reg:gamma`, `count:poisson`, and `reg:tweedie` transformations
  (Exp + `base_score` log conversion).

### [scikit-learn](https://github.com/scikit-learn/scikit-learn) (experimental)

- Loads `pickle` (protocol `0`) trees.
- `sklearn.ensemble.GradientBoostingClassifier` supported.

### Training (v2.x, milestones T1–T5)

- `gbtree` / `dart` / `gblinear`; `hist` / `exact` / `gpu_hist` (`auto` selects
  by row count).
- Objectives: regression, binary / multiclass, gamma, poisson, tweedie,
  `survival:cox` / `survival:aft`, and ranking (`rank:ndcg` /
  `rank:pairwise` / `rank:listwise`).
- Cross-validation, early stopping, checkpointing, resume, monotonic
  constraints, `max_leaves` (lossguide), and `BatchedMatrix` for
  out-of-core DMatrix.
- Exporter: native `leaves.json` and **XGBoost 3.x JSON**.

### Deployment & observability

- int8 threshold quantization (with a parity gate).
- `Ensemble.Reload` for hot model rotation; per-batch inference profiling.
- [WASM demo](examples/wasm/README.md) (js / wasm, Native CPU),
  [HTTP embed demo](examples/http/README.md) (batched serving),
  and [train-from-model demo](examples/train_from_model/README.md).

## Installation

```sh
go install github.com/linkerlin/leaves/v2/cmd/leaves@latest
```

Module path: `github.com/linkerlin/leaves/v2` (Go 1.26+).
Tensors / GPU acceleration are powered by
[github.com/born-ml/born](https://github.com/born-ml/born).

## Quick start

The simplest path: load with `leaves.LoadFromFile` and call `PredictSingle`.

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
	fmt.Printf("prediction for %v: %f\n", fvals, p)
}
```

Legacy aliases still work: `leaves.LGEnsembleFromFile(path, loadTransformation)`
and `leaves.XGEnsembleFromFile(...)`. Their behaviour is unchanged.

### Format auto-detection and support tiers

`io.LoadFromFile` / `DetectFormat` use extension + header probe. Full support
tiers (stable / experimental / placeholder): [docs/interop-matrix.md](docs/interop-matrix.md).

| Format | Tier | Extension / probe | Notes |
| ------ | ---- | ----------------- | ----- |
| leaves.json | **stable** | `leaves_version` | Primary train artifact; preferred deploy format |
| XGBoost JSON | **stable** | `.json` + `learner` | 3.x default; logistic `base_score` → margin |
| XGBoost UBJSON | **stable** | `.ubj` | Predictions match JSON path |
| XGBoost binary | **stable** | `binf` / header | Prefer JSON/UBJ for new work |
| LightGBM text/JSON | **stable** | `tree=` / `tree_info` | Both text and JSON |
| scikit-learn | **experimental** | `.pkl` / `.joblib` | Narrow protocol; export to XGB/leaves JSON for production |
| ONNX | **experimental** | `.onnx` | TreeEnsembleRegressor subset; convert complex graphs first |

Load failures return `*io.LoadError` with an actionable `hint:`. Numeric tables
misnamed as `.txt` models point you to `data.FromFile`.

```go
import "github.com/linkerlin/leaves/v2/io"

m, err := io.LoadFromFile("model.ubj", io.DefaultLoadOptions()) // AutoTransform = true

m, err = io.LoadFromFile("model.ubj", &io.LoadOptions{
	AutoTransform:      false, // raw margin output
	LoadTransformation: true,  // or force transforms for every objective
	Backend:            io.BackendAuto,
})
```

When `AutoTransform` and `LoadTransformation` are both `false`, `OutputValue`
and `OutputMargin` are both raw margin. `contrib` / SHAP are *always* in
margin space regardless of `AutoTransform`.

## Training API

```go
dm, _ := leaves.LoadDataAuto("train.tsv") // content sniff

learner, _ := leaves.NewLearner(leaves.TrainConfig{
	Objective: leaves.TrainObjectiveBinaryLogistic,
	NumRound:  50,
})
_ = learner.Fit(dm)
_ = learner.Save("out.leaves.json")
```

### Out-of-core DMatrix

```go
bm, _ := data.SplitDense(dense, batchRows) // or data.NewBatchedMatrix(...)
learner, _ := train.FitExternal(train.Config{
	Objective:     train.ObjectiveBinaryLogistic,
	TreeMethod:    train.TreeMethodHist,
	HistBinPolicy: "global",
	NumRound:      10,
}, bm)
```

### Checkpoint resume

```go
learner, _ := train.ResumeFit("ckpt.json", train.Config{
	Objective:  train.ObjectiveSquaredError,
	NumRound:   20, // continues past the checkpointed rounds
	TreeMethod: train.TreeMethodExact,
}, dm)
score, _ := learner.Eval(dm)
```

### Training data — auto-sniffing

`data.FromFileAuto` / `leaves.LoadDataAuto` reads a sample of the file and
selects a loader; the extension is only a fallback.

| Sniffed format     | Typical content                | Loader            |
| ------------------ | ------------------------------ | ----------------- |
| LIBSVM             | `label 1:0.5 3:1.2`            | `FromLIBSVM`      |
| Ranking TSV        | `qid label feat...`            | `LoadRankingTSV`  |
| TSV, label-last    | No header, label in last column | `LoadDenseTSV`  |
| CSV                | Header or heuristic label col  | `FromCSV`         |

### Infer the objective from a reference model

```go
learner, _ := leaves.NewLearnerFromModelAndData(
	"reference_model.json", // reads objective only (XGB JSON / UBJ / leaves.json)
	"train.tsv",            // LoadDataAuto content sniff
	leaves.TrainConfig{NumRound: 10, TreeMethod: leaves.TrainTreeMethodExact},
	data.DefaultFileLoadOptions(),
)
_ = learner.Save("out.leaves.json")
```

```bash
go run ./examples/train_from_model/ -model testdata/xgboost_smoke.json -data testdata/breast_cancer_train.tsv
```

### `survival:aft` interval censoring

```go
d, _ := data.NewDense(vals, n, nfeat, placeholderLabels, nil)
dm, _ := data.NewAFTDense(d, []data.AFTInterval{
	{1.5, 1.5},              // event
	{2, math.Inf(1)},        // right-censored
	{0, 4},                  // left-censored
	{1, 6},                  // interval-censored
})
learner, _ := train.NewLearner(train.Config{Objective: train.ObjectiveSurvivalAFT, ...})
_ = learner.Fit(dm)
```

### Monotonic constraints

```go
learner, _ := train.NewLearner(train.Config{
	Objective:           "reg:squarederror",
	MonotoneConstraints: []int{1, 0, -1}, // feature 0 increasing, 2 decreasing
	TreeMethod:          train.TreeMethodHist,
})
```

### Acceleration stack (T4+)

The hist / `gpu_hist` path is layered (log prefix `[leaves/train] accel:`):

| Stage            | Implementation                                          | Default `auto` (≥30k rows + WebGPU) | `LEAVES_TRAIN_ACCEL=webgpu` |
| ---------------- | ------------------------------------------------------- | ----------------------------------- | --------------------------- |
| Global binning   | Per-feature cut points + row-level bin cache at fit time | ✅                                  | ✅                           |
| Histogram accum. | CPU `rowBin` / WebGPU `SelectAdd` batch                  | `gpu_hist`                          | GPU batch                    |
| Gain scan        | Pure CPU / batched WebGPU gain                           | Batched GPU (GPU path)              | Batched GPU                  |
| Margin predict   | Row-by-row CPU / Born GPU `PredictDense`                 | `PredictMargins` ≥ 256 rows         | ≥ 64 rows                    |

```go
learner, _ := train.NewLearner(train.Config{
	Objective:     train.ObjectiveRankNDCG,
	TreeMethod:    train.TreeMethodGPUHist,
	AccelMode:     train.AccelModeWebGPU, // or env LEAVES_TRAIN_ACCEL
	NumThreads:    4,
	HistBinPolicy: "global", // default; "per_node" restores per-node cuts
	NumRound:      50,
})
_ = learner.Fit(dm)
```

Environment override: `LEAVES_TRAIN_ACCEL=auto|webgpu|born_cpu|cpu`.

## Compute backend — [Born](https://github.com/born-ml/born)

Inference and training acceleration sit on
[Born](https://github.com/born-ml/born) (CPU SIMD + WebGPU). `NativeEngine`
is the golden reference; **BackendAuto 2.0** dispatches BornCPU / BornGPU by
workload. Full decision table: [docs/backend-auto.md](docs/backend-auto.md).

```go
m, _ := leaves.LoadFromFile("model.json", &io.LoadOptions{
	Backend:  io.BackendAuto,
	Workload: tree.WorkloadHint{BatchSize: 256, HasGPU: true},
})
// Explained: tree.SelectBackendExplained(caps, hint) → Rule / Reason
```

The parity gate `TestBornParityFormatMatrix` covers LGB text/JSON, XGB
bin/json/ubj, scikit-learn pickle × batch `{1, 16, 256}` × `BornCPU` /
`BornGPU` at tolerance `1e-5` relative to Native.

### Backend selection cheat sheet (BackendAuto 2.0)

| Scenario | Auto result | Notes |
| -------- | ----------- | ----- |
| Online / batch &lt; 64 | **Native** | Latency-first; Born is slow at batch=1 |
| Batch ≥ 64, pure numeric | **BornCPU** | SIMD batch path |
| Batch ≥ 256 + HasGPU (Windows WebGPU) | **BornGPU** | Falls back to BornCPU if unavailable |
| WASM + numeric | **BornCPU** when supported | No GPU; cat-small → Native |
| Sparse (`SparseDensity∈(0,0.15)`) / cat-small | **Native** | Not Born-optimized / unsupported |

## Tree SHAP and explainability

`m.Explain()` exposes Lundberg fast Tree SHAP (with `SumHess` weight
coverage, `tree_path_dependent`); SHAP values are in margin space, matching
`OutputMargin` / `predict.OutputContribution`. `DefaultLoadOptions()` returns
*probabilities* for logistic objectives; contrib / SHAP remain in margin
space.

```go
import (
	"github.com/linkerlin/leaves/v2/v2"
	"github.com/linkerlin/leaves/v2/explain"
)

m, _ := leaves.LoadFromFile("model.json", leaves.DefaultLoadOptions())

x := [][]float64{{1.0, 2.0, 3.0}}
contrib, _ := m.Explain().TreeSHAP(x)        // margin-space SHAP
inter, _ := m.Explain().InteractionSHAP(x)   // interaction matrix
mc, _ := m.Explain().TreeSHAPMulticlass(x)   // [sample][feature][class]
base := m.Explain().ExpectedValue()          // baseline (all-zero features)

imp := m.Explain().Importance(explain.ImportanceGain, nil)
text := m.Explain().DumpText(nil)
dot := m.Explain().DumpDOT(nil)
```

The unified output path is [`predict.Request`](predict/request.go):

```go
import "github.com/linkerlin/leaves/v2/predict"

nf := m.NFeatures()
nRows := 1

// value (transformed output for logistic, raw for squared error)
vals := make([]float64, nRows*m.NOutputGroups())
_ = m.PredictWithRequest(predict.Request{
	Matrix: predict.DenseMatrix{Values: flat, Rows: nRows, Cols: nf},
	Output: predict.OutputValue,
}, vals)

// raw margin (before sigmoid / softmax)
margins := make([]float64, nRows*m.NRawOutputGroups())
_ = m.PredictWithRequest(predict.Request{
	Matrix: predict.DenseMatrix{Values: flat, Rows: nRows, Cols: nf},
	Output: predict.OutputMargin,
}, margins)

// leaf indices (int32 per tree)
nTrees := m.NTrees()
leaves := make([]int32, nRows*nTrees)
_ = m.PredictWithRequest(predict.Request{
	Matrix: predict.DenseMatrix{Values: flat, Rows: nRows, Cols: nf},
	Output: predict.OutputLeaf,
}, leaves)

// SHAP contributions (margin space; binary: [sample][feature+bias])
out := make([]float64, (nf+1)*nRows)
_ = m.PredictWithRequest(predict.Request{
	Matrix: predict.DenseMatrix{Values: flat, Rows: nRows, Cols: nf},
	Output: predict.OutputContribution, // or OutputApproxContribution / OutputInteraction
}, out)
```

`bias` semantics: the trailing column is the **background margin** at
all-zero features, matching the additivity convention of XGBoost's
`pred_contribs` (per-element decomposition may differ).

## Metrics — `metrics/`

Built-in RMSE / MAE / AUC / LogLoss / MAPE / RMSLE / NDCG@k / MAP, named to
match XGBoost's `eval_metric`:

```go
import "github.com/linkerlin/leaves/v2/metrics"

rmse, _ := metrics.RMSE{}.Evaluate(yTrue, yPred)
m, _ := metrics.Resolve("ndcg@5", metrics.Options{Groups: []int{10, 10}})
ndcg, _ := m.Evaluate(yTrue, yPred)
```

Ranking metrics need a `data.GroupedMatrix` exposing `Groups()`.

## Learning to rank

XGBoost-compatible LambdaMART plus a native listwise head:

| Objective        | Family                                       | Notes                                                                       |
| ---------------- | -------------------------------------------- | --------------------------------------------------------------------------- |
| `rank:ndcg`      | LambdaMART (XGBoost-compatible)              | Pairwise with \|ΔNDCG\| weighting; targets the ranking metric directly.      |
| `rank:pairwise`  | RankNet pairwise                             | Defaults to **top-k** (k = 32) + `lambdarank_normalization` (XGB-aligned).  |
| `rank:listwise`  | **ListNet softmax CE** (leaves native)       | `q ∝ exp(label)`, `p = softmax(pred)`, pure listwise cross-entropy.         |

```go
import (
	"github.com/linkerlin/leaves/v2/data"
	"github.com/linkerlin/leaves/v2/train"
)

dm, _ := data.LoadRankingTSV("rank_train.tsv", "\t") // qid label feat1 feat2 ...

learner, _ := train.NewLearner(train.Config{
	Objective:    train.ObjectiveRankNDCG, // or ObjectiveRankPairwise / ObjectiveRankListwise
	NumRound:     40,
	MaxDepth:     4,
	LearningRate: 0.1,
	NDCGK:        10,
	EvalMetric:   "ndcg@10",
})
_ = learner.Fit(dm)
```

End-to-end MovieLens 100K demo (train → save → top-K recommend):
[`demos/movielens/README.md`](demos/movielens/README.md).

```bash
cd testdata && python gen_rank_movielens.py
go run ./demos/movielens/cmd/train
go run ./demos/movielens/cmd/recommend -group 0 -topk 10
go test ./train/... -run TestRankMovieLens -v
```

## Callbacks and LR schedulers

```go
learner, _ := train.NewLearner(train.Config{
	Objective:    train.ObjectiveSquaredError,
	NumRound:     20,
	LearningRate: 0.3,
	LRScheduler:  train.ExponentialLRScheduler(0.95), // ×0.95 per round
	Callbacks: []train.TrainingCallback{
		train.FuncCallback(func(ctx *train.CallbackContext) error {
			// ctx.Round, ctx.LearningRate, ctx.TrainMetric, ctx.EvalMetric
			return nil
		}),
	},
})
```

Built-in schedulers: `ExponentialLRScheduler(gamma)` and
`StepLRScheduler(every, factor)`. `CallbackContext` also fills
`EvalMetric` / `EvalMetricOK` when `EvalSet` + `EvalMetric` are configured
(no `EarlyStop` required).

## Inference profiling and hot reload

```go
if ne, ok := m.Engine().(*tree.NativeEngine); ok {
	prof, err := tree.ProfileNativeDense(ne, vals, nrows, ncols, preds, 0)
	_ = prof.Elapsed
}

// Atomic model rotation in production
_ = m.Reload("/path/to/new.model", io.DefaultLoadOptions())
```

`tree.ProfileWalkStats` reports per-sample tree-walk depth without running
the full prediction.

## int8 threshold quantization

Numerical split thresholds are int8-affine quantised per feature
(127 levels); **leaf values stay float64** and categorical nodes are not
quantised. `quantize.Engine` does not support `PredictLeafIndices*`
(margin only).

```go
qf, _ := quantize.QuantizeForest(m.Forest(), quantize.Config{})
res, err := quantize.CheckParityWithGate(m.Forest(), qf, samples, 0, quantize.DefaultGate())

eng, _ := quantize.NewEngine(qf, nil, tree.TransformRaw, m.NOutputGroups())
model.NewEnsemble(eng) // swap the live engine
```

## Agent automation (SKILL-driven, no MCP)

Agents drive `train → tune → publish` through **SKILL + the `leaves` CLI +
metrics.json**; the **agent is the optimizer** (search strategy lives in the
SKILL text, the library ships no built-in HPO).

### Three steps to let an agent train for you

1. **Install the CLI**: `go install github.com/linkerlin/leaves/v2/cmd/leaves@latest`
   (or use `go run ./cmd/leaves ...` inside the repo).
2. **Give the agent the skill**: Cursor picks up `.cursor/skills` out of the
   box; other agents read [`skills/leaves-autotrain/SKILL.md`](skills/leaves-autotrain/SKILL.md).
3. **Say one sentence** (no flags to learn):

```text
Use the leaves-autotrain skill; training data is data/train.csv,
drive RMSE down, publish to release/v1 once converged.
```

The agent sniffs the data to pick the objective, builds a CV baseline, tunes
via the SKILL §4.5 evolutionary search protocol (Hall-of-Fame + fold-level
Pareto parent selection, reflective mutation from `--emit-rounds` curves and
`explain` importances, ledger lineage tags in `runs.jsonl`, budget cap), then
`publish`es the artifact bundle (model + manifest + reproduce script). No Go
code is written anywhere in the loop.

Contract details and DoD: [演进方案.md](演进方案.md) (v2.0). CLI flags and
metrics schema: [`skills/leaves-autotrain/cli.md`](skills/leaves-autotrain/cli.md).
Zero-prep demo: [`examples/autotrain/`](examples/autotrain/README.md).

## Documentation

| Document | Description |
| -------- | ----------- |
| [godoc](https://pkg.go.dev/github.com/linkerlin/leaves/v2) | API reference |
| [docs/api-surface.md](docs/api-surface.md) | Recommended / compat / experimental APIs |
| [docs/versioning.md](docs/versioning.md) | What can change in v2.x |
| [docs/release-checklist.md](docs/release-checklist.md) | Pre-tag release checklist |
| [docs/interop-matrix.md](docs/interop-matrix.md) | Format support tiers |
| [docs/backend-auto.md](docs/backend-auto.md) | BackendAuto 2.0 decision table |
| [docs/extension-points.md](docs/extension-points.md) | Custom objective/metric |
| [CHANGELOG.md](CHANGELOG.md) | Release history |
| [演进计划.md](演进计划.md) | Library 12-month roadmap (v5.4) |
| [演进方案.md](演进方案.md) | Agentic contract (done) |
| [NOTES.md](NOTES.md) | Live compat notes (AutoTransform, etc.) |
| [compatibility.md](compatibility.md) | External GBRT correctness matrix |
| [AGENTS.md](AGENTS.md) | Project conventions |
| [examples/](examples/) | autotrain / wasm / http / train_from_model |

## Compatibility

- API layers & migration: [docs/api-surface.md](docs/api-surface.md)
- Version policy: [docs/versioning.md](docs/versioning.md)
- Behaviour notes: [NOTES.md](NOTES.md)
- External library versions: [compatibility.md](compatibility.md)

## Performance

Single-thread, batch 1000, on a 2017 15" MacBook Pro (2.9 GHz Core i7,
16 GB 2133 MHz LPDDR3). C-API numbers come from Python bindings; per-call
overhead is negligible in the batched regime. `go test -bench` drives the
go side; see [benchmark/](benchmark/) and
[testdata/README.md](testdata/README.md) for data prep.

Single-thread:

| Case                                                              | Features | Trees | Batch | C API | _leaves_ |
| ----------------------------------------------------------------- | -------- | ----- | ----- | ----- | -------- |
| LightGBM [MS LTR](https://github.com/microsoft/LightGBM/blob/master/docs/Experiments.rst#comparison-experiment) | 137      | 500   | 1000  | 49 ms | 51 ms    |
| LightGBM [Higgs](https://github.com/microsoft/LightGBM/blob/master/docs/Experiments.rst#comparison-experiment)   | 28       | 500   | 1000  | 50 ms | 50 ms    |
| LightGBM KDD Cup 99*                                              | 41       | 1200  | 1000  | 70 ms | 85 ms    |
| XGBoost Higgs                                                     | 28       | 500   | 1000  | 44 ms | 50 ms    |

Four threads:

| Case                                                              | Features | Trees | Batch | C API | _leaves_ |
| ----------------------------------------------------------------- | -------- | ----- | ----- | ----- | -------- |
| LightGBM [MS LTR](https://github.com/microsoft/LightGBM/blob/master/docs/Experiments.rst#comparison-experiment) | 137      | 500   | 1000  | 14 ms | 14 ms    |
| LightGBM [Higgs](https://github.com/microsoft/LightGBM/blob/master/docs/Experiments.rst#comparison-experiment)   | 28       | 500   | 1000  | 14 ms | 14 ms    |
| LightGBM KDD Cup 99*                                              | 41       | 1200  | 1000  | 19 ms | 24 ms    |
| XGBoost Higgs                                                     | 28       | 500   | 1000  | ?     | 14 ms    |

(`?`) XGBoost multithreaded prediction is not reachable through the
Python bindings. (`*`) KDD Cup 99 mixes continuous and categorical
features.

For a Born-backend microbench (Windows / WebGPU) see
[docs/benchmark-baseline.md](docs/benchmark-baseline.md).

## Limitations

- **LightGBM** — limited set of transformations (sigmoid and softmax only).
- **XGBoost** — limited set of transformations (sigmoid and softmax only);
  floating-point rounding means C-API and `_leaves_` predictions may differ
  by a handful of ULPs.
- **scikit-learn** — no transformation support (output is the raw score from
  `GradientBoostingClassifier.decision_function`); only pickle protocol `0`;
  same float-rounding caveat as XGBoost.

## Roadmap

The roadmap now focuses on the current codebase audit and the next 12
months of work after P0–T5 / v3.1 were delivered. See
[演进计划.md](演进计划.md) for the **v5.4** plan, and [TODO.md](TODO.md)
for the closed historical backlog.

## Contact

For questions or collaboration, email **steper@foxmail.com**.

## License

[MIT](LICENSE.md) — Copyright (c) 2018 Steper Lin.
