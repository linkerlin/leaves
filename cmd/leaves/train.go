package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/linkerlin/leaves/v2/booster"
	"github.com/linkerlin/leaves/v2/data"
	"github.com/linkerlin/leaves/v2/train"
)

// modelTreeCount 返回 learner 当前模型的树数（EVO-02；gblinear/无森林为 0）。
func modelTreeCount(l *train.Learner) int {
	ir := l.Model()
	if ir == nil || ir.Forest == nil {
		return 0
	}
	return len(ir.Forest.Trees)
}

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
	naPolicy := fs.String("na-policy", "error", "缺失值策略：error（默认）| skip-row（丢行）| nan（转为 NaN 走树默认方向，WP-31）")
	autoEncode := fs.Bool("auto-encode", false, "将非数值字符串特征列自动按字典序编码并持久化进模型（WP-31）")
	strictFlags := fs.Bool("strict-flags", false, "严格模式：--cv 与 --val/--emit-rounds 并存时失败（默认仅警告）")
	labelCol := fs.String("label-col", "", "显式指定标签列名或 0-based 索引（WP-22）")
	dropCols := fs.String("drop-cols", "", "逗号分隔需丢弃的列名或索引（如 ID、时间戳）")
	boosterName := fs.String("booster", "gbtree", "基学习器：gbtree（默认）| dart | gblinear（WP-27）")
	rateDrop := fs.Float64("rate-drop", 0.1, "DART 树丢弃率（--booster dart 时生效）")
	skipDrop := fs.Float64("skip-drop", 0.0, "DART 跳过丢弃的前 N 棵树")
	monotone := fs.String("monotone", "", "单调性约束，逗号分隔 -1/0/1（如 1,0,-1）")
	checkpointPath := fs.String("checkpoint", "", "checkpoint JSON 输出路径")
	checkpointEvery := fs.Int("checkpoint-every", 0, "每 N 轮写出一次 checkpoint")
	resumePath := fs.String("resume", "", "从已有 checkpoint 恢复续训路径")
	batchRows := fs.Int("batch-rows", 0, "外存分批训练每批行数（>0 启用 BatchedMatrix）")
	lrDecay := fs.Float64("lr-decay", 0, "指数学习率衰减因子 gamma（0=不衰减）")
	weightCol := fs.String("weight-col", "", "样本权重列名或 0-based 索引（WP-41）")
	scalePosWeight := fs.Float64("scale-pos-weight", 0, "二分类正样本（y>0.5）梯度/Hessian 权重乘子（WP-41）")
	tweediePower := fs.Float64("tweedie-variance-power", 0, "reg:tweedie 方差幂参数 (1,2)，默认 1.5（WP-42）")
	multiOutputTree := fs.Bool("multi-output-tree", false, "多分类/多目标使用向量叶单树（WP-42）")
	numParallelTree := fs.Int("num-parallel-tree", 0, "每轮并行随机森林树数（默认 1，WP-42）")
	dryRun := fs.Bool("dry-run", false, "仅校验参数、数据、标签门禁与模型配置，不执行训练循环（WP-29）")
	if err := fs.Parse(args); err != nil {
		return err
	}
	// WP-17：--from-run 填未显式设置的旋钮；显式 CLI flag 始终优先。
	if *fromRun != "" {
		rec, notice, err := loadRunFromLedger(*fromRun, *tag)
		if err != nil {
			return err
		}
		if notice != "" {
			fmt.Fprintf(os.Stderr, "leaves: 注意：%s\n", notice)
		}
		set := map[string]bool{}
		fs.Visit(func(f *flag.Flag) { set[f.Name] = true })
		applyParamsIfUnset(set, rec.Params, objective, evalMetric,
			numClass, numTarget, rounds, depth, maxLeaves,
			lr, lambda, minChildWeight, gamma,
			maxBin, subsample, colsample, treeMethod,
			ndcgK, cv, earlyStop, seed, rec.Objective)
		applyExtraParamsIfUnset(set, rec, rounds, valPath, cv,
			labelCol, dropCols, boosterName, rateDrop, skipDrop, monotone, lrDecay,
			batchRows, naPolicy, autoEncode)
		applyApexParamsIfUnset(set, rec, weightCol, scalePosWeight, tweediePower, multiOutputTree, numParallelTree)
		if *tag == "" && rec.Tag != "" {
			// 未指定 tag：沿用源 tag 加后缀，避免账本混淆同名
			*tag = rec.Tag + "_repro"
		}
		fmt.Fprintf(os.Stderr, "leaves: --from-run 已加载 tag=%q value=%g（CLI 显式 flag 优先）\n", rec.Tag, rec.Value)
	}
	if *resumePath != "" && *objective == "" {
		if _, obj, _, cerr := train.LoadCheckpointFile(*resumePath); cerr == nil && obj != "" {
			*objective = obj
		}
	}
	if *dataPath == "" || *objective == "" {
		return errUsage("--data 与 --objective 必需（--objective 也可由 --from-run 账本或 --resume 提供）")
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
	// WP-24：--cv 与 --early-stop 支持折内早停协同；仅 --cv 与 --val / --emit-rounds 互斥。
	if *cv >= 2 && (*valPath != "" || *emitRounds != "") {
		msg := "--cv 与 --val/--emit-rounds 互斥：cv 路径忽略后者"
		hint := "基线用 --cv（可配 --early-stop 做折内早停）；独立验证集或逐轮诊断去掉 --cv 改用 --val"
		if *strictFlags {
			return errAgent("cv_conflict", msg, hint, false)
		}
		fmt.Fprintf(os.Stderr, "leaves: 注意：%s（要强制失败请加 --strict-flags）\n", msg)
	}
	// 自动建输出父目录（与 publish 一致，免 Agent 踩坑）。
	for _, p := range []string{filepath.Dir(*outModel), filepath.Dir(*outFinal), filepath.Dir(*metricsPath), filepath.Dir(*runsPath), filepath.Dir(*emitRounds), filepath.Dir(*checkpointPath)} {
		if p != "" && p != "." {
			_ = os.MkdirAll(p, 0o755)
		}
	}

	dm, err := loadMatrixFull(*dataPath, *naPolicy, *numTarget, *labelCol, *dropCols, "", *weightCol, *autoEncode, nil, nil)
	if err != nil {
		return err
	}
	if err := requireLabelColumn(dm, *dataPath, *numTarget); err != nil {
		return err
	}
	var trainCatMaps map[string]map[string]float64
	if d, ok := dm.(*data.Dense); ok {
		trainCatMaps = d.CategoryMaps
	} else if dwg, ok := dm.(*data.DenseWithGroups); ok && dwg.Dense != nil {
		trainCatMaps = dwg.Dense.CategoryMaps
	}
	trainFeatNames := featureNamesOf(dm)

	monoVec, err := parseMonotoneConstraints(*monotone)
	if err != nil {
		return errUsage(fmt.Sprintf("--monotone: %v", err))
	}

	bKind := strings.ToLower(strings.TrimSpace(*boosterName))
	var cfgBooster string
	var dartCfg *booster.DARTConfig
	switch bKind {
	case "", "gbtree":
		cfgBooster = train.BoosterGBTree
	case "dart":
		cfgBooster = train.BoosterGBTree
		dartCfg = &booster.DARTConfig{
			RateDrop:      *rateDrop,
			SkipDrop:      int(*skipDrop),
			NormalizeType: "tree",
		}
	case "gblinear":
		cfgBooster = train.BoosterGBLinear
	default:
		return errUsage(fmt.Sprintf("未知 --booster %q（支持 gbtree|dart|gblinear）", *boosterName))
	}

	cfg := train.Config{
		Booster:              cfgBooster,
		Objective:            *objective,
		NumTarget:            *numTarget,
		NumRound:             *rounds,
		MaxDepth:             *depth,
		MaxLeaves:            *maxLeaves,
		LearningRate:         *lr,
		Lambda:               *lambda,
		MinHessian:           *minChildWeight,
		Gamma:                *gamma,
		MaxBin:               *maxBin,
		Subsample:            *subsample,
		ColsampleByTree:      *colsample,
		TreeMethod:           *treeMethod,
		Seed:                 *seed,
		EvalMetric:           metric,
		NDCGK:                *ndcgK,
		MonotoneConstraints:  monoVec,
		CheckpointPath:       *checkpointPath,
		CheckpointEvery:      *checkpointEvery,
		DART:                 dartCfg,
		ScalePosWeight:       *scalePosWeight,
		TweedieVariancePower: *tweediePower,
		MultiOutputTree:      *multiOutputTree,
		NumParallelTree:      *numParallelTree,
	}
	if *lrDecay > 0 {
		cfg.LRScheduler = train.ExponentialLRScheduler(*lrDecay)
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
	if *labelCol != "" {
		doc.Params.LabelCol = *labelCol
	}
	if *dropCols != "" {
		doc.Params.DropCols = *dropCols
	}
	if bKind != "" && bKind != "gbtree" {
		doc.Params.Booster = bKind
		if bKind == "dart" {
			doc.Params.RateDrop = *rateDrop
			doc.Params.SkipDrop = *skipDrop
		}
	}
	if *monotone != "" {
		doc.Params.Monotone = *monotone
	}
	if *lrDecay > 0 {
		doc.Params.LRDecay = *lrDecay
	}
	if *batchRows > 0 {
		doc.Params.BatchRows = *batchRows
	}
	if *naPolicy != "" && *naPolicy != "error" {
		doc.Params.NAPolicy = *naPolicy
	}
	if *autoEncode {
		doc.Params.AutoEncode = true
	}
	if *weightCol != "" {
		doc.Params.WeightCol = *weightCol
	}
	if *scalePosWeight > 0 {
		doc.Params.ScalePosWeight = *scalePosWeight
	}
	if *tweediePower > 0 {
		doc.Params.TweedieVariancePower = *tweediePower
	}
	if *multiOutputTree {
		doc.Params.MultiOutputTree = true
	}
	if *numParallelTree > 0 {
		doc.Params.NumParallelTree = *numParallelTree
	}
	if *valPath != "" && *cv < 2 {
		doc.Params.Val = *valPath
	}

	if *dryRun {
		if _, err := train.NewLearner(cfg); err != nil {
			return err
		}
		return writeJSON(*metricsPath, map[string]any{
			"schema_version": metricsSchemaVersion,
			"dry_run":        true,
			"valid":          true,
			"objective":      *objective,
			"metric":         metric,
			"maximize":       doc.Maximize,
			"n_rows":         dm.NumRow(),
			"n_features":     dm.NumCol(),
			"params":         doc.Params,
		})
	}

	// 交叉验证路径：CV 出诚实估计；支持折内早停；若要存模型再在全量上单跑一次。
	trainStart := time.Now()
	if *cv >= 2 {
		if *earlyStop > 0 {
			cfg.EarlyStop = train.NewEarlyStopping(*earlyStop, doc.Maximize)
		}
		res, err := train.CrossValidate(cfg, dm, *cv)
		if err != nil {
			return fmt.Errorf("cv: %w", err)
		}
		doc.CVFolds = *cv
		doc.CVMean = res.MeanMetric
		doc.CVStd = res.StdMetric
		doc.FoldMetrics = res.FoldMetrics
		doc.Value = res.MeanMetric
		if len(res.FoldBestRounds) > 0 {
			doc.CVBestRounds = res.FoldBestRounds
			doc.CVBestRoundMean = res.MeanBestRound
			doc.BestRound = int(math.Round(res.MeanBestRound))
		}
		if *outModel != "" {
			fullCfg := cfg
			fullCfg.EarlyStop = nil
			fullCfg.EvalSet = nil
			if doc.BestRound > 0 {
				fullCfg.NumRound = doc.BestRound
			}
			full, err := train.NewLearner(fullCfg)
			if err != nil {
				return err
			}
			if err := full.Fit(dm); err != nil {
				return err
			}
			doc.TrainAccel = full.EffectiveAccelMode()
			doc.ModelRound = full.BoostRounds()
			if err := full.Save(*outModel); err != nil {
				return err
			}
			doc.NTrees = modelTreeCount(full)
		}
		doc.ElapsedMS = time.Since(trainStart).Milliseconds()
		if err := writeMetrics(*metricsPath, doc); err != nil {
			return err
		}
		return appendRun(*runsPath, *tag, *outModel, doc)
	}

	// 单次训练路径。
	var valDM data.Matrix
	if *valPath != "" {
		valDM, err = loadMatrixFull(*valPath, *naPolicy, *numTarget, *labelCol, *dropCols, "", *weightCol, *autoEncode, trainCatMaps, trainFeatNames)
		if err != nil {
			return err
		}
		if err := requireLabelColumn(valDM, *valPath, *numTarget); err != nil {
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

	fitDM := dm
	if *batchRows > 0 {
		if d, ok := dm.(*data.Dense); ok {
			if cfg.TreeMethod == "" || cfg.TreeMethod == "auto" {
				cfg.TreeMethod = train.TreeMethodHist
			}
			bm, err := data.SplitDense(d, *batchRows)
			if err != nil {
				return err
			}
			fitDM = bm
		}
	}

	var learner *train.Learner
	if *resumePath != "" {
		learner, err = train.LoadCheckpoint(*resumePath, cfg)
		if err != nil {
			return err
		}
	} else {
		learner, err = train.NewLearner(cfg)
		if err != nil {
			return err
		}
	}
	if err := learner.Fit(fitDM); err != nil {
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
	doc.NTrees = modelTreeCount(learner)
	doc.ElapsedMS = time.Since(trainStart).Milliseconds()

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

	diagDM := dm
	if valDM != nil {
		diagDM = valDM
	}
	if *objective == "binary:logistic" {
		diagMargins := make([]float64, diagDM.NumRow())
		if err := learner.PredictMargins(diagDM, diagMargins); err == nil {
			probs := make([]float64, len(diagMargins))
			for i, m := range diagMargins {
				probs[i] = sigmoid(m)
			}
			doc.BinaryDiagnostics = computeBinaryDiagnostics(diagDM.Labels(), probs)
		}
	} else if isMulticlassObjective(*objective) && *numClass >= 2 {
		diagMargins := make([]float64, diagDM.NumRow()*(*numClass))
		if err := learner.PredictMargins(diagDM, diagMargins); err == nil {
			doc.MulticlassDiagnostics = computeMulticlassDiagnostics(diagDM.Labels(), diagMargins, *numClass)
		}
	} else if *numTarget < 2 && (strings.HasPrefix(*objective, "reg:") || strings.HasPrefix(*objective, "count:")) {
		diagMargins := make([]float64, diagDM.NumRow())
		if err := learner.PredictMargins(diagDM, diagMargins); err == nil {
			preds := metricInputs("rmse", *objective, diagMargins, 1)
			doc.RegressionDiagnostics = computeRegressionDiagnostics(diagDM.Labels(), preds)
		}
	} else if strings.HasPrefix(*objective, "rank:") {
		if grps := groupsOf(diagDM); len(grps) > 0 {
			diagMargins := make([]float64, diagDM.NumRow())
			if err := learner.PredictMargins(diagDM, diagMargins); err == nil {
				doc.RankingDiagnostics = computeRankingDiagnostics(diagDM.Labels(), diagMargins, grps)
			}
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

func parseMonotoneConstraints(spec string) ([]int, error) {
	spec = strings.TrimSpace(spec)
	if spec == "" {
		return nil, nil
	}
	spec = strings.TrimPrefix(strings.TrimSuffix(spec, ")"), "(")
	spec = strings.TrimPrefix(strings.TrimSuffix(spec, "]"), "[")
	parts := strings.Split(spec, ",")
	out := make([]int, 0, len(parts))
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		v, err := strconv.Atoi(p)
		if err != nil || (v != -1 && v != 0 && v != 1) {
			return nil, fmt.Errorf("invalid monotone constraint %q (must be -1, 0, or 1)", p)
		}
		out = append(out, v)
	}
	return out, nil
}
