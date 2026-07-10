package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"

	"github.com/linkerlin/leaves/data"
	"github.com/linkerlin/leaves/train"
)

func cmdTrain(args []string) error {
	fs := flag.NewFlagSet("train", flag.ContinueOnError)
	fs.Usage = func() { fmt.Fprintln(fs.Output(), "用法: leaves train --data PATH --objective NAME [flags]") }
	dataPath := fs.String("data", "", "训练数据路径（必需，自动嗅探）")
	objective := fs.String("objective", "", "目标函数（必需）")
	evalMetric := fs.String("eval-metric", "", "评估指标（默认按 objective）")
	numClass := fs.Int("num-class", 0, "多分类类数（multi:* 必需）")
	rounds := fs.Int("rounds", 50, "boosting 轮数")
	depth := fs.Int("depth", 6, "最大深度")
	maxLeaves := fs.Int("max-leaves", 0, "lossguide 叶数（0=depthwise）")
	lr := fs.Float64("lr", 0.3, "学习率")
	lambda := fs.Float64("lambda", 1.0, "L2")
	minChildWeight := fs.Float64("min-child-weight", 1.0, "min_child_weight（MinHessian）")
	gamma := fs.Float64("gamma", 0, "分裂最小增益")
	maxBin := fs.Int("max-bin", 256, "hist 最大桶数")
	subsample := fs.Float64("subsample", 1.0, "行采样")
	colsample := fs.Float64("colsample", 1.0, "列采样")
	treeMethod := fs.String("tree-method", "auto", "hist|exact|auto")
	ndcgK := fs.Int("ndcg-k", 10, "排序 NDCG@k")
	cv := fs.Int("cv", 0, "K 折交叉验证（0=不交叉验证）")
	valPath := fs.String("val", "", "独立验证集路径")
	earlyStop := fs.Int("early-stop", 0, "N 轮无改进早停（建议配 --val）")
	seed := fs.Int64("seed", 42, "随机种子")
	outModel := fs.String("out-model", "", "输出 leaves.json 路径")
	metricsPath := fs.String("metrics", "", "输出 metrics.json（空=stdout）")
	runsPath := fs.String("runs", "", "运行账本 JSONL（追加本次记录，Agent 优化记忆）")
	tag := fs.String("tag", "", "本次运行标签（写入账本，便于回溯）")
	emitRounds := fs.String("emit-rounds", "", "逐轮指标 JSONL（Agent 学习曲线诊断用）")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *dataPath == "" || *objective == "" {
		return errUsage("--data 与 --objective 必需")
	}
	if (*objective == "multi:softmax" || *objective == "multi:softprob") && *numClass < 2 {
		return errUsage("multi:* 目标需要 --num-class >= 2")
	}
	metric := *evalMetric
	if metric == "" {
		metric = defaultMetric(*objective)
	}
	if *cv >= 2 && (*valPath != "" || *earlyStop > 0 || *emitRounds != "") {
		fmt.Fprintln(os.Stderr, "leaves: 注意：--cv 路径不使用 --val/--early-stop/--emit-rounds（后者被忽略）；要逐轮诊断请去掉 --cv 改用 --val")
	}
	// 自动建输出父目录（与 publish 一致，免 Agent 踩坑）。
	for _, p := range []string{filepath.Dir(*outModel), filepath.Dir(*metricsPath), filepath.Dir(*runsPath), filepath.Dir(*emitRounds)} {
		if p != "" && p != "." {
			_ = os.MkdirAll(p, 0o755)
		}
	}

	dm, err := data.FromFileAuto(*dataPath)
	if err != nil {
		return fmt.Errorf("load train: %w", err)
	}

	cfg := train.Config{
		Objective:       *objective,
		NumRound:        *rounds,
		MaxDepth:        *depth,
		MaxLeaves:       *maxLeaves,
		LearningRate:    *lr,
		Lambda:          *lambda,
		MinHessian:      *minChildWeight,
		Gamma:           *gamma,
		MaxBin:          *maxBin,
		Subsample:       *subsample,
		ColsampleByTree: *colsample,
		TreeMethod:      *treeMethod,
		Seed:            *seed,
		EvalMetric:      metric,
		NDCGK:           *ndcgK,
	}
	if *numClass > 0 {
		cfg.NumClass = *numClass
	}

	nc := cfg.NumClass
	if nc < 1 {
		nc = 1
	}
	doc := metricsDoc{
		Objective: *objective,
		Metric:    metric,
		NRows:     dm.NumRow(),
		Maximize:  metricMaximize(metric, nc, groupsOf(dm)),
		Params: &paramsRecord{
			Rounds: *rounds, Depth: *depth, MaxLeaves: *maxLeaves,
			LR: *lr, Lambda: *lambda, TreeMethod: *treeMethod, Seed: *seed,
		},
	}

	// 交叉验证路径：CV 出诚实估计；若要存模型再在全量上单跑一次。
	if *cv >= 2 {
		res, err := train.CrossValidate(cfg, dm, *cv)
		if err != nil {
			return fmt.Errorf("cv: %w", err)
		}
		doc.CVFolds = *cv
		doc.CVMean = res.MeanMetric
		doc.CVStd = res.StdMetric
		doc.FoldMetrics = res.FoldMetrics
		doc.Value = res.MeanMetric
		if *outModel != "" {
			full, err := train.NewLearner(cfg)
			if err != nil {
				return err
			}
			if err := full.Fit(dm); err != nil {
				return err
			}
			if err := full.Save(*outModel); err != nil {
				return err
			}
		}
		if err := writeMetrics(*metricsPath, doc); err != nil {
			return err
		}
		return appendRun(*runsPath, *tag, *outModel, doc)
	}

	// 单次训练路径。
	var valDM data.Matrix
	if *valPath != "" {
		valDM, err = data.FromFileAuto(*valPath)
		if err != nil {
			return fmt.Errorf("load val: %w", err)
		}
		cfg.EvalSet = valDM
		if *earlyStop > 0 {
			cfg.EarlyStop = train.NewEarlyStopping(*earlyStop, doc.Maximize)
		}
	}

	// 逐轮指标（Agent 学习曲线诊断）：仅单次训练路径，CV 不支持。
	if *emitRounds != "" {
		rf, err := os.Create(*emitRounds)
		if err != nil {
			return fmt.Errorf("emit-rounds: %w", err)
		}
		defer rf.Close()
		enc := json.NewEncoder(rf)
		cfg.Callbacks = append(cfg.Callbacks, train.FuncCallback(func(ctx *train.CallbackContext) error {
			rec := map[string]any{"round": ctx.Round, "lr": ctx.LearningRate}
			if ctx.TrainMetricOK {
				rec["train"] = ctx.TrainMetric
			}
			if ctx.EvalMetricOK {
				rec["val"] = ctx.EvalMetric
			}
			return enc.Encode(rec)
		}))
	}

	learner, err := train.NewLearner(cfg)
	if err != nil {
		return err
	}
	if err := learner.Fit(dm); err != nil {
		return err
	}

	trainScore, err := learner.Eval(dm)
	if err != nil {
		return fmt.Errorf("eval train: %w", err)
	}
	doc.TrainMetric = trainScore
	doc.Value = trainScore
	if valDM != nil {
		vScore, err := learner.Eval(valDM)
		if err != nil {
			return fmt.Errorf("eval val: %w", err)
		}
		doc.Value = vScore
	}
	if cfg.EarlyStop != nil {
		doc.BestRound = cfg.EarlyStop.BestRound()
		if doc.BestRound > 0 {
			doc.Value = cfg.EarlyStop.BestScore // 报告 best-round val，而非 final-round（过拟合）val
		}
	}

	if *outModel != "" {
		if err := learner.Save(*outModel); err != nil {
			return err
		}
	}
	if err := writeMetrics(*metricsPath, doc); err != nil {
		return err
	}
	return appendRun(*runsPath, *tag, *outModel, doc)
}
