package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/linkerlin/leaves/data"
	"github.com/linkerlin/leaves/train"
)

func cmdTrain(args []string) error {
	fs := flag.NewFlagSet("train", flag.ContinueOnError)
	fs.Usage = func() {
		fmt.Fprintln(fs.Output(), "用法: leaves train --data PATH --objective NAME [flags]")
		fmt.Fprintln(fs.Output(), "      leaves train --data PATH --from-run runs.jsonl [--tag NAME] [覆盖 flags]")
	}
	dataPath := fs.String("data", "", "训练数据路径（必需，自动嗅探）")
	objective := fs.String("objective", "", "目标函数（必需；可由 --from-run 账本补全）")
	evalMetric := fs.String("eval-metric", "", "评估指标（默认按 objective）")
	numClass := fs.Int("num-class", 0, "多分类类数（multi:* 必需）")
	numTarget := fs.Int("num-target", 0, "多目标回归目标维（≥2；CSV 末 N 列为标签；one_output_per_tree）")
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
	outModel := fs.String("out-model", "", "输出 leaves.json 路径（早停+save-best 时为 best_round）")
	outFinal := fs.String("out-final", "", "早停时另存 final-round 模型（在截断前写出；POST-12）")
	metricsPath := fs.String("metrics", "", "输出 metrics.json（空=stdout）")
	runsPath := fs.String("runs", "", "运行账本 JSONL（追加本次记录，Agent 优化记忆）")
	tag := fs.String("tag", "", "本次运行标签（写入账本；与 --from-run 联用时兼作选行键）")
	emitRounds := fs.String("emit-rounds", "", "逐轮指标 JSONL（Agent 学习曲线诊断用）")
	saveBest := fs.Bool("save-best", true, "早停时截断并保存 best_round 模型（默认 true；false=保留 final-round）")
	fromRun := fs.String("from-run", "", "从 runs.jsonl 加载 params 作默认（CLI 覆盖优先；无 --tag 则取最优行）")
	naPolicy := fs.String("na-policy", "error", "缺失值策略：error（默认，遇空/NA 失败）| skip-row（丢弃含缺失的整行）")
	strictFlags := fs.Bool("strict-flags", false, "严格模式：--cv 与 --val/--early-stop/--emit-rounds 并存时失败（默认仅警告）")
	if err := fs.Parse(args); err != nil {
		return err
	}
	// WP-17：--from-run 填未显式设置的旋钮；显式 CLI flag 始终优先。
	if *fromRun != "" {
		rec, err := loadRunFromLedger(*fromRun, *tag)
		if err != nil {
			return err
		}
		set := map[string]bool{}
		fs.Visit(func(f *flag.Flag) { set[f.Name] = true })
		// 选中账本行后，本次新 run 的 --tag 若未另给，保留选中 tag 便于追溯；
		// 但用户若未传 --tag，from-run 选最优时 tag 已是最优行的 tag——不覆盖用户新 tag。
		// 注意：tag 在 load 时已用于选行，这里不再改 *tag。
		applyParamsIfUnset(set, rec.Params, objective, evalMetric,
			numClass, numTarget, rounds, depth, maxLeaves,
			lr, lambda, minChildWeight, gamma,
			maxBin, subsample, colsample, treeMethod,
			ndcgK, cv, earlyStop, seed, rec.Objective)
		if *tag == "" && rec.Tag != "" {
			// 未指定 tag：沿用源 tag 加后缀，避免账本混淆同名
			*tag = rec.Tag + "_repro"
		}
		fmt.Fprintf(os.Stderr, "leaves: --from-run 已加载 tag=%q value=%g（CLI 显式 flag 优先）\n", rec.Tag, rec.Value)
	}
	if *dataPath == "" || *objective == "" {
		return errUsage("--data 与 --objective 必需（--objective 也可由 --from-run 账本提供）")
	}
	// 防 Agent 踩坑：同一路径会先 Save 再被 metrics 覆盖，inspect 必失败。
	if *outModel != "" && *metricsPath != "" && filepath.Clean(*outModel) == filepath.Clean(*metricsPath) {
		return errUsage("--out-model 与 --metrics 不能是同一路径（metrics 会覆盖模型文件）")
	}
	if *outFinal != "" && *outModel != "" && filepath.Clean(*outFinal) == filepath.Clean(*outModel) {
		return errUsage("--out-final 与 --out-model 不能是同一路径")
	}
	if *outFinal != "" && *metricsPath != "" && filepath.Clean(*outFinal) == filepath.Clean(*metricsPath) {
		return errUsage("--out-final 与 --metrics 不能是同一路径")
	}
	if (*objective == "multi:softmax" || *objective == "multi:softprob") && *numClass < 2 {
		return errAgent("objective_mismatch", "multi:* 目标需要 --num-class >= 2",
			"先 leaves sniff --data ... 取 label.n_unique 作为 --num-class", false)
	}
	if *numTarget >= 2 {
		if strings.HasPrefix(*objective, "multi:") || strings.HasPrefix(*objective, "rank:") {
			return errAgent("objective_mismatch", "--num-target 不能与 multi:*/rank:* 同用",
				"多目标回归用 reg:squarederror 等 + --num-target N", false)
		}
	}
	metric := *evalMetric
	if metric == "" {
		metric = defaultMetric(*objective)
	}
	if *cv >= 2 && (*valPath != "" || *earlyStop > 0 || *emitRounds != "") {
		msg := "--cv 与 --val/--early-stop/--emit-rounds 互斥：cv 路径忽略后者"
		hint := "基线用 --cv；逐轮诊断/早停去掉 --cv 改用 --val --early-stop；Agent 建议加 --strict-flags"
		if *strictFlags {
			return errAgent("cv_conflict", msg, hint, false)
		}
		fmt.Fprintf(os.Stderr, "leaves: 注意：%s（要强制失败请加 --strict-flags）\n", msg)
	}
	// 自动建输出父目录（与 publish 一致，免 Agent 踩坑）。
	for _, p := range []string{filepath.Dir(*outModel), filepath.Dir(*outFinal), filepath.Dir(*metricsPath), filepath.Dir(*runsPath), filepath.Dir(*emitRounds)} {
		if p != "" && p != "." {
			_ = os.MkdirAll(p, 0o755)
		}
	}

	dm, err := loadMatrixOpts(*dataPath, *naPolicy, *numTarget)
	if err != nil {
		return err
	}

	cfg := train.Config{
		Objective:       *objective,
		NumTarget:       *numTarget,
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
	ndcgKParam := 0
	if strings.HasPrefix(*objective, "rank:") {
		ndcgKParam = *ndcgK
	}
	doc := metricsDoc{
		Objective: *objective,
		Metric:    metric,
		NRows:     dm.NumRow(),
		NFeatures: dm.NumCol(),
		Maximize:  metricMaximize(metric, nc, groupsOf(dm)),
		Params: newParamsRecord(
			*rounds, *depth, *maxLeaves,
			*lr, *lambda, *minChildWeight, *gamma,
			*maxBin, *subsample, *colsample,
			*treeMethod, *seed,
			*numClass, *numTarget, ndcgKParam, *earlyStop, *cv,
			metric,
		),
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
			doc.TrainAccel = full.EffectiveAccelMode()
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
		valDM, err = loadMatrixOpts(*valPath, *naPolicy, *numTarget)
		if err != nil {
			return err
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
	doc.TrainAccel = learner.EffectiveAccelMode()

	// 早停后：记录实际训练轮数，可选截断到 best_round 再 Save（WP-03）。
	// POST-12：--out-final 在截断前写出 final-round 侧车。
	doc.StoppedRound = learner.BoostRounds()
	if cfg.EarlyStop != nil {
		doc.BestRound = cfg.EarlyStop.BestRound()
		if doc.BestRound > 0 {
			doc.Value = cfg.EarlyStop.BestScore // 报告 best-round val，而非 final-round
		}
		if *outFinal != "" {
			if err := learner.Save(*outFinal); err != nil {
				return fmt.Errorf("save final model: %w", err)
			}
			doc.FinalModel = *outFinal
			doc.FinalRound = learner.BoostRounds()
		}
		if *saveBest && doc.BestRound > 0 {
			learner.ApplyBestRound()
		}
	} else if *outFinal != "" {
		// 无早停：final 与主模型同内容；仍写出便于 Agent 固定路径读取。
		if err := learner.Save(*outFinal); err != nil {
			return fmt.Errorf("save final model: %w", err)
		}
		doc.FinalModel = *outFinal
		doc.FinalRound = learner.BoostRounds()
	}
	doc.ModelRound = learner.BoostRounds()

	trainScore, err := learner.Eval(dm)
	if err != nil {
		return fmt.Errorf("eval train: %w", err)
	}
	doc.TrainMetric = trainScore
	if valDM == nil && (cfg.EarlyStop == nil || doc.BestRound <= 0) {
		doc.Value = trainScore
	}
	if valDM != nil && (cfg.EarlyStop == nil || doc.BestRound <= 0) {
		vScore, err := learner.Eval(valDM)
		if err != nil {
			return fmt.Errorf("eval val: %w", err)
		}
		doc.Value = vScore
	}
	// 有 early-stop 时 value 已是 BestScore；截断后 val 应与之对齐（可选复核，不覆盖 BestScore）。

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
