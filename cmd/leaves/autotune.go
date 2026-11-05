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
)

// cmdAutotune 实现有界预算反射式演化自治闭环（WP-43 / DoD AA4）：
// 自动串联 sniff（脏数据自适应）→ baseline → diagnose → runs propose/crossover → 最优模型定稿。
func cmdAutotune(args []string) error {
	fs := flag.NewFlagSet("autotune", flag.ContinueOnError)
	fs.Usage = func() {
		fmt.Fprintln(fs.Output(), "用法: leaves autotune --data PATH [--val PATH] [--cv K] [--budget N] [--out-dir DIR] [--out-model PATH] [--metrics PATH]")
	}
	dataPath := fs.String("data", "", "训练数据路径（必需）")
	valPath := fs.String("val", "", "可选独立验证集路径")
	objective := fs.String("objective", "", "目标函数（空=由 sniff 自动推断）")
	evalMetric := fs.String("eval-metric", "", "评估指标（空=由 sniff 自动推断）")
	numClass := fs.Int("num-class", 0, "多分类类别数（空=由 sniff 自动推断）")
	numTarget := fs.Int("num-target", 0, "多目标回归目标维（≥2）")
	cv := fs.Int("cv", 0, "K 折交叉验证（无 --val 且未设置时默认 3）")
	budget := fs.Int("budget", 4, "有界演化训练总轮次预算（1..30，默认 4）")
	rounds := fs.Int("rounds", 30, "基线 boosting 轮数（默认 30）")
	earlyStop := fs.Int("early-stop", 0, "早停耐心轮数")
	labelCol := fs.String("label-col", "", "显式指定标签列名或 0-based 索引")
	dropCols := fs.String("drop-cols", "", "逗号分隔需丢弃的列名或索引")
	weightCol := fs.String("weight-col", "", "样本权重列名或 0-based 索引")
	naPolicy := fs.String("na-policy", "", "缺失值策略：error|skip-row|nan（空=按 sniff 自动适配）")
	autoEncode := fs.Bool("auto-encode", false, "将非数值字符串特征列自动按字典序编码（sniff 检测到类别列时自动开启）")
	outDir := fs.String("out-dir", "", "演化工作目录（默认 autotune_out 或 --out-model 所在目录）")
	outModel := fs.String("out-model", "", "最优定稿模型输出路径（默认 <out-dir>/best.leaves.json）")
	metricsPath := fs.String("metrics", "", "输出 autotune 报告 JSON 路径（空=stdout 并在 <out-dir>/autotune_report.json 留存）")
	taskNameFlag := fs.String("task", "", "任务名称/签名（用于检索与沉淀跨任务 lessons 记忆；默认取数据集文件名，WP-48）")
	recordLesson := fs.Bool("record-lesson", true, "当演化优于基线时自动将最优超参经验沉淀入 lessons 记忆库（默认 true，WP-48）")
	seed := fs.Int64("seed", 42, "随机种子")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *dataPath == "" {
		return errUsage("leaves autotune 需要 --data PATH")
	}
	if *budget < 1 {
		*budget = 1
	}
	if *budget > 30 {
		*budget = 30
	}

	workDir := strings.TrimSpace(*outDir)
	if workDir == "" {
		if *outModel != "" {
			workDir = filepath.Dir(*outModel)
		} else {
			workDir = "autotune_out"
		}
	}
	if err := os.MkdirAll(workDir, 0o755); err != nil {
		return fmt.Errorf("mkdir %s: %w", workDir, err)
	}

	finalModelPath := strings.TrimSpace(*outModel)
	if finalModelPath == "" {
		finalModelPath = filepath.Join(workDir, "best.leaves.json")
	}

	// Step 0: Sniff 数据画像与脏数据自适应
	effObj := strings.TrimSpace(*objective)
	effMetric := strings.TrimSpace(*evalMetric)
	effNumClass := *numClass
	effDrops := strings.TrimSpace(*dropCols)
	effNA := strings.TrimSpace(*naPolicy)
	effAutoEncode := *autoEncode
	var scalePosWeight float64

	if *numTarget < 2 {
		rep, err := buildSniffReport(*dataPath, "nan", *labelCol, effDrops, true)
		if err != nil {
			return err
		}
		if effObj == "" {
			if s, ok := rep["suggested_objective"].(string); ok && s != "" {
				effObj = s
			}
		}
		if effMetric == "" {
			if s, ok := rep["suggested_metric"].(string); ok && s != "" {
				effMetric = s
			}
		}
		if (effObj == "multi:softmax" || effObj == "multi:softprob") && effNumClass < 2 {
			if lm, ok := rep["label"].(map[string]any); ok && lm != nil {
				if nu, ok := asFloat(lm["n_unique"]); ok && int(nu) >= 2 {
					effNumClass = int(nu)
				}
			}
		}
		if cats, ok := rep["categorical_columns"].([]string); ok && len(cats) > 0 {
			effAutoEncode = true
		}
		if sdrops, ok := rep["suggested_drop_cols"].([]string); ok && len(sdrops) > 0 {
			effDrops = mergeDropCols(effDrops, sdrops)
		}
		if dq, ok := rep["data_quality"].(map[string]any); ok && dq != nil {
			if nCells, ok := dq["nan_cells"].(int); ok && nCells > 0 && effNA == "" {
				effNA = "nan"
			}
			if cf, ok := dq["constant_features"].([]string); ok && len(cf) > 0 {
				effDrops = mergeDropCols(effDrops, cf)
			}
			if effObj == "binary:logistic" {
				if posRatio, ok := asFloat(dq["label_pos_ratio"]); ok && posRatio > 0 && posRatio < 1 {
					if posRatio < 0.25 || posRatio > 0.75 {
						spw := math.Round(((1.0-posRatio)/posRatio)*100) / 100
						if spw > 0 && spw != 1.0 {
							scalePosWeight = spw
						}
					}
				}
			}
		}
	}
	if effObj == "" {
		effObj = "reg:squarederror"
	}
	if effMetric == "" {
		effMetric = defaultMetric(effObj)
	}
	if effNA == "" {
		effNA = "error"
	}
	effCV := *cv
	if effCV == 0 && *valPath == "" && *numTarget < 2 {
		effCV = 3
	}

	taskSig := strings.TrimSpace(*taskNameFlag)
	if taskSig == "" {
		baseName := filepath.Base(*dataPath)
		ext := filepath.Ext(baseName)
		taskSig = strings.TrimSuffix(baseName, ext)
		if taskSig == "" {
			taskSig = "autotune_task"
		}
	}
	priorLessons, _ := searchLessonsRecords(taskSig+" "+effObj+" "+effMetric, 5)

	runsPath := filepath.Join(workDir, "runs.jsonl")
	_ = os.Remove(runsPath) // 每次 autotune 在工作目录内建立干净账本

	// Step 1: Baseline 训练
	baseModelPath := filepath.Join(workDir, "step_1_baseline.leaves.json")
	baseMetricsPath := filepath.Join(workDir, "step_1_metrics.json")
	baseRoundsPath := filepath.Join(workDir, "step_1_rounds.jsonl")

	baseArgs := []string{
		"--data", *dataPath,
		"--objective", effObj,
		"--eval-metric", effMetric,
		"--rounds", strconv.Itoa(*rounds),
		"--depth", "6",
		"--lr", "0.2",
		"--seed", strconv.FormatInt(*seed, 10),
		"--na-policy", effNA,
		"--out-model", baseModelPath,
		"--metrics", baseMetricsPath,
		"--runs", runsPath,
		"--tag", "baseline",
	}
	if effNumClass >= 2 {
		baseArgs = append(baseArgs, "--num-class", strconv.Itoa(effNumClass))
	}
	if *numTarget >= 2 {
		baseArgs = append(baseArgs, "--num-target", strconv.Itoa(*numTarget))
	}
	if *labelCol != "" {
		baseArgs = append(baseArgs, "--label-col", *labelCol)
	}
	if effDrops != "" {
		baseArgs = append(baseArgs, "--drop-cols", effDrops)
	}
	if *weightCol != "" {
		baseArgs = append(baseArgs, "--weight-col", *weightCol)
	}
	if effAutoEncode {
		baseArgs = append(baseArgs, "--auto-encode")
	}
	if scalePosWeight > 0 {
		baseArgs = append(baseArgs, "--scale-pos-weight", strconv.FormatFloat(scalePosWeight, 'f', -1, 64))
	}
	if effCV >= 2 {
		baseArgs = append(baseArgs, "--cv", strconv.Itoa(effCV))
		if *earlyStop > 0 {
			baseArgs = append(baseArgs, "--early-stop", strconv.Itoa(*earlyStop))
		}
	} else {
		if *valPath != "" {
			baseArgs = append(baseArgs, "--val", *valPath)
		}
		if *earlyStop > 0 {
			baseArgs = append(baseArgs, "--early-stop", strconv.Itoa(*earlyStop))
		}
		baseArgs = append(baseArgs, "--emit-rounds", baseRoundsPath)
	}

	if err := cmdTrain(baseArgs); err != nil {
		return fmt.Errorf("autotune baseline train: %w", err)
	}

	lastMetricsPath := baseMetricsPath
	lastModelPath := baseModelPath
	lastRoundsPath := ""
	if effCV < 2 {
		lastRoundsPath = baseRoundsPath
	}

	// Step 2..Budget: 反射式演化循环（diagnose -> propose / crossover -> train）
	for step := 2; step <= *budget; step++ {
		records, err := readAllRuns(runsPath)
		if err != nil || len(records) == 0 {
			break
		}
		diagPath := filepath.Join(workDir, fmt.Sprintf("step_%d_diag.json", step))
		diagArgs := []string{
			"--metrics", lastMetricsPath,
			"--model", lastModelPath,
			"--data", *dataPath,
			"--out", diagPath,
		}
		if lastRoundsPath != "" {
			diagArgs = append(diagArgs, "--rounds", lastRoundsPath)
		}
		if *labelCol != "" {
			diagArgs = append(diagArgs, "--label-col", *labelCol)
		}
		if effDrops != "" {
			diagArgs = append(diagArgs, "--drop-cols", effDrops)
		}
		_ = cmdDiagnose(diagArgs)

		planPath := filepath.Join(workDir, fmt.Sprintf("step_%d_plan.json", step))
		useCrossover := step >= 3 && len(records) >= 2 && step%3 == 0
		if useCrossover {
			err = cmdRunsCrossover([]string{
				"--runs", runsPath,
				"--seed", strconv.FormatInt(*seed+int64(step), 10),
				"--metrics", planPath,
			})
		} else {
			propArgs := []string{
				"--runs", runsPath,
				"--seed", strconv.FormatInt(*seed+int64(step), 10),
				"--metrics", planPath,
			}
			if _, serr := os.Stat(diagPath); serr == nil {
				propArgs = append(propArgs, "--diagnose", diagPath)
			}
			err = cmdRunsPropose(propArgs)
		}
		if err != nil {
			return fmt.Errorf("autotune step %d plan: %w", step, err)
		}

		planBytes, err := os.ReadFile(planPath)
		if err != nil {
			return err
		}
		var planDoc map[string]any
		if err := json.Unmarshal(planBytes, &planDoc); err != nil {
			return err
		}

		var cliArgs []string
		if rawArgs, ok := planDoc["suggested_cli_args"].([]any); ok {
			for _, a := range rawArgs {
				if s, ok := a.(string); ok {
					cliArgs = append(cliArgs, s)
				}
			}
		}
		stepModelPath := filepath.Join(workDir, fmt.Sprintf("step_%d_model.leaves.json", step))
		stepMetricsPath := filepath.Join(workDir, fmt.Sprintf("step_%d_metrics.json", step))
		stepRoundsPath := filepath.Join(workDir, fmt.Sprintf("step_%d_rounds.jsonl", step))

		trainArgs := []string{"--data", *dataPath}
		trainArgs = append(trainArgs, cliArgs...)
		if *valPath != "" && effCV < 2 {
			trainArgs = append(trainArgs, "--val", *valPath)
		}
		if effCV < 2 {
			trainArgs = append(trainArgs, "--emit-rounds", stepRoundsPath)
		}
		trainArgs = append(trainArgs,
			"--seed", strconv.FormatInt(*seed, 10),
			"--out-model", stepModelPath,
			"--metrics", stepMetricsPath,
			"--runs", runsPath,
		)
		if err := cmdTrain(trainArgs); err != nil {
			return fmt.Errorf("autotune step %d train: %w", step, err)
		}
		lastMetricsPath = stepMetricsPath
		lastModelPath = stepModelPath
		if effCV < 2 {
			lastRoundsPath = stepRoundsPath
		}
	}

	// Step 3: 汇总账本并定稿最优模型
	allRuns, err := readAllRuns(runsPath)
	if err != nil || len(allRuns) == 0 {
		return fmt.Errorf("autotune: empty runs ledger after execution")
	}
	baseRun := allRuns[0]
	bestRun := allRuns[0]
	lineage := make([]map[string]any, len(allRuns))
	for i, r := range allRuns {
		if betterRun(&r, &bestRun) {
			bestRun = r
		}
		lineage[i] = map[string]any{
			"step":       i + 1,
			"tag":        r.Tag,
			"value":      r.Value,
			"n_trees":    r.NTrees,
			"elapsed_ms": r.ElapsedMS,
			"model":      r.Model,
		}
	}

	if bestRun.Model != "" && filepath.Clean(bestRun.Model) != filepath.Clean(finalModelPath) {
		if err := copyFile(bestRun.Model, finalModelPath); err != nil {
			return fmt.Errorf("copy best model: %w", err)
		}
	}

	var impPct float64
	if math.Abs(baseRun.Value) > 1e-12 {
		if bestRun.Maximize {
			impPct = ((bestRun.Value - baseRun.Value) / math.Abs(baseRun.Value)) * 100.0
		} else {
			impPct = ((baseRun.Value - bestRun.Value) / math.Abs(baseRun.Value)) * 100.0
		}
	}

	var distilledLesson *lessonRecord
	if *recordLesson && bestRun.Tag != baseRun.Tag && impPct > 0 && bestRun.Params != nil {
		p := bestRun.Params
		rec := lessonRecord{
			Task:     fmt.Sprintf("%s (%s)", taskSig, effObj),
			Lesson:   fmt.Sprintf("最优变异 %s：depth=%d, lr=%g, rounds=%d, lambda=%g, min_child_weight=%g", bestRun.Tag, p.Depth, p.LR, p.Rounds, p.Lambda, p.MinChildWeight),
			Evidence: fmt.Sprintf("%s: %.4f -> %.4f (%+.2f%% via %s)", effMetric, baseRun.Value, bestRun.Value, impPct, bestRun.Tag),
			Tags:     []string{effObj, effMetric, "autotune"},
		}
		if saved, lerr := appendLessonRecord(rec); lerr == nil {
			distilledLesson = &saved
		}
	}

	report := map[string]any{
		"schema_version":  metricsSchemaVersion,
		"task":            taskSig,
		"objective":       effObj,
		"metric":          effMetric,
		"maximize":        bestRun.Maximize,
		"budget":          *budget,
		"runs_executed":   len(allRuns),
		"baseline_tag":    baseRun.Tag,
		"baseline_value":  baseRun.Value,
		"best_tag":        bestRun.Tag,
		"best_value":      bestRun.Value,
		"improvement_pct": round4(impPct),
		"best_model":      finalModelPath,
		"runs_ledger":     runsPath,
		"best_params":     bestRun.Params,
		"auto_adaptations": map[string]any{
			"auto_encode":      effAutoEncode,
			"na_policy":        effNA,
			"drop_cols":        effDrops,
			"scale_pos_weight": scalePosWeight,
			"cv_folds":         effCV,
		},
		"lineage": lineage,
	}
	if len(priorLessons) > 0 {
		report["prior_lessons"] = priorLessons
	}
	if distilledLesson != nil {
		report["distilled_lesson"] = distilledLesson
	}

	defaultReportPath := filepath.Join(workDir, "autotune_report.json")
	_ = writeJSON(defaultReportPath, report)
	if *metricsPath != "" && filepath.Clean(*metricsPath) != filepath.Clean(defaultReportPath) {
		return writeJSON(*metricsPath, report)
	}
	if *metricsPath == "" {
		return writeJSON("", report)
	}
	return nil
}
