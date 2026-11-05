package main

import (
	"flag"
	"fmt"
	"math"
	"math/rand"
	"sort"
	"strings"
	"time"

	"github.com/linkerlin/leaves/v2/data"
	leavesio "github.com/linkerlin/leaves/v2/io"
	"github.com/linkerlin/leaves/v2/metrics"
)

type compareModelEntry struct {
	Model         string     `json:"model"`
	Format        string     `json:"format,omitempty"`
	Objective     string     `json:"objective"`
	Metric        string     `json:"metric"`
	Value         float64    `json:"value"`
	NTrees        int        `json:"n_trees"`
	InferUS       int64      `json:"infer_us"`
	DeltaVsBase   float64    `json:"delta_vs_baseline"`
	DeltaPct      float64    `json:"delta_pct"`
	Better        bool       `json:"better_than_baseline"`
	WinRate       float64    `json:"bootstrap_win_rate,omitempty"`
	ImprovementCI [2]float64 `json:"bootstrap_improvement_ci95,omitempty"`
	Significant   bool       `json:"significant"`
}

func cmdCompare(args []string) error {
	fs := flag.NewFlagSet("compare", flag.ContinueOnError)
	fs.Usage = func() {
		fmt.Fprintln(fs.Output(), "用法: leaves compare --models M1,M2[,...] --data PATH [flags]")
	}
	modelsSpec := fs.String("models", "", "逗号分隔的模型路径列表（至少 2 个，首项为 baseline；支持 leaves.json/XGB/LGB/ONNX/.quant.json）")
	dataPath := fs.String("data", "", "评估数据路径（必需）")
	evalMetric := fs.String("eval-metric", "", "评估指标（默认按首个模型 objective 推断）")
	objective := fs.String("objective", "", "目标函数（默认按首个模型推断）")
	bootstrapN := fs.Int("bootstrap", 200, "配对 Bootstrap 重采样次数（0=不计算显著性；默认 200）")
	seed := fs.Int64("seed", 42, "Bootstrap 随机种子")
	numTarget := fs.Int("num-target", 0, "多目标评估：CSV 末 N 列标签")
	labelCol := fs.String("label-col", "", "显式指定标签列名或 0-based 索引")
	dropCols := fs.String("drop-cols", "", "逗号分隔需丢弃的列名或索引")
	naPolicy := fs.String("na-policy", "error", "缺失值策略：error|skip-row|nan")
	autoEncode := fs.Bool("auto-encode", false, "将非数值字符串特征列自动按字典序编码")
	metricsPath := fs.String("metrics", "", "输出对比报告 JSON（空=stdout）")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if strings.TrimSpace(*modelsSpec) == "" || *dataPath == "" {
		return errUsage("--models 与 --data 必需")
	}

	var modelPaths []string
	for _, p := range strings.Split(*modelsSpec, ",") {
		p = strings.TrimSpace(p)
		if p != "" {
			modelPaths = append(modelPaths, p)
		}
	}
	if len(modelPaths) < 2 {
		return errUsage("--models 至少需要 2 个模型路径（逗号分隔）")
	}

	obj := *objective
	var catMaps map[string]map[string]float64
	var expectedFeats []string
	for _, mp := range modelPaths {
		if obj == "" {
			if _, irObj, _, err := leavesio.LoadModelIR(mp); err == nil && irObj != "" {
				obj = irObj
			}
		}
		if len(catMaps) == 0 {
			catMaps = loadModelCategoryMaps(mp)
		}
		if len(expectedFeats) == 0 {
			expectedFeats = loadModelFeatureNames(mp)
		}
	}
	if obj == "" {
		obj = "reg:squarederror"
	}
	metric := *evalMetric
	if metric == "" {
		metric = defaultMetric(obj)
	}

	firstEns, err := loadEnsemble(modelPaths[0])
	if err != nil {
		return fmt.Errorf("load model %s: %w", modelPaths[0], err)
	}
	nc := firstEns.NOutputGroups()
	firstEns.Close()
	if nc < 1 {
		nc = 1
	}
	nt := *numTarget
	if nt < 2 && nc > 1 && !isMulticlassObjective(obj) {
		nt = nc
	}

	dm, err := loadMatrixFull(*dataPath, *naPolicy, nt, *labelCol, *dropCols, "", "", *autoEncode, catMaps, expectedFeats)
	if err != nil {
		return err
	}
	if err := requireLabelColumn(dm, *dataPath, nt); err != nil {
		return err
	}
	vals, err := denseVals(dm)
	if err != nil {
		return err
	}

	groups := groupsOf(dm)
	mt, err := metrics.Resolve(metric, metrics.Options{NumClass: nc, Groups: groups})
	if err != nil {
		return fmt.Errorf("resolve metric %q: %w", metric, err)
	}
	maximize := mt.HigherIsBetter()

	labels := dm.Labels()
	if nt >= 2 {
		if mtm, ok := data.AsMultiTarget(dm); ok {
			labels = mtm.Targets()
		}
	}

	nRows := dm.NumRow()
	nCols := dm.NumCol()
	entries := make([]compareModelEntry, len(modelPaths))
	allMetricPreds := make([][]float64, len(modelPaths))

	for idx, mp := range modelPaths {
		m, err := loadEnsemble(mp)
		if err != nil {
			return fmt.Errorf("load model %s: %w", mp, err)
		}
		nTrees := len(m.NLeaves())
		if nTrees <= 0 {
			nTrees = m.NEstimators() * m.NRawOutputGroups()
		}
		fmtName := "quant"
		if !strings.HasSuffix(mp, ".quant.json") {
			if _, _, f, ierr := leavesio.LoadModelIR(mp); ierr == nil {
				fmtName = leavesio.SupportOf(f).Name
			}
		}

		preds := make([]float64, nRows*nc)
		t0 := time.Now()
		predErr := m.PredictDense(vals, nRows, nCols, preds, 0, 0)
		inferUS := time.Since(t0).Microseconds()
		m.Close()
		if predErr != nil {
			return fmt.Errorf("predict %s: %w", mp, predErr)
		}

		mPreds := metricInputs(metric, obj, preds, nc)
		if nt >= 2 && !isMulticlassObjective(obj) {
			mPreds = preds
		}
		score, err := metrics.Evaluate(mt, labels, mPreds, groups)
		if err != nil {
			return fmt.Errorf("evaluate %s: %w", mp, err)
		}
		allMetricPreds[idx] = mPreds
		entries[idx] = compareModelEntry{
			Model:     mp,
			Format:    fmtName,
			Objective: obj,
			Metric:    metric,
			Value:     score,
			NTrees:    nTrees,
			InferUS:   inferUS,
		}
	}

	baseScore := entries[0].Value
	basePreds := allMetricPreds[0]
	winnerIdx := 0

	for idx := 0; idx < len(entries); idx++ {
		if idx > 0 {
			diff := entries[idx].Value - baseScore
			entries[idx].DeltaVsBase = diff
			if math.Abs(baseScore) > 1e-12 {
				entries[idx].DeltaPct = (diff / math.Abs(baseScore)) * 100.0
			}
			if maximize {
				entries[idx].Better = entries[idx].Value > baseScore
			} else {
				entries[idx].Better = entries[idx].Value < baseScore
			}
			if *bootstrapN > 0 && len(groups) == 0 && nt < 2 {
				winRate, ciLow, ciHigh, sig := pairedBootstrap(
					mt, maximize, labels, basePreds, allMetricPreds[idx], nRows, nc, *bootstrapN, *seed+int64(idx),
				)
				entries[idx].WinRate = round4(winRate)
				entries[idx].ImprovementCI = [2]float64{round4(ciLow), round4(ciHigh)}
				entries[idx].Significant = sig
			}
		}
		if maximize {
			if entries[idx].Value > entries[winnerIdx].Value {
				winnerIdx = idx
			}
		} else {
			if entries[idx].Value < entries[winnerIdx].Value {
				winnerIdx = idx
			}
		}
	}

	report := map[string]any{
		"schema_version": metricsSchemaVersion,
		"data":           *dataPath,
		"n_rows":         nRows,
		"n_features":     nCols,
		"objective":      obj,
		"metric":         metric,
		"maximize":       maximize,
		"baseline_model": modelPaths[0],
		"winner_model":   modelPaths[winnerIdx],
		"winner_index":   winnerIdx,
		"models":         entries,
	}
	return writeJSON(*metricsPath, report)
}

// pairedBootstrap 执行确定性配对 Bootstrap 重采样，返回候选优于基线的胜率与改进量 95% CI。
func pairedBootstrap(
	mt metrics.Metric,
	maximize bool,
	labels, basePreds, candPreds []float64,
	nRows, nc, iters int,
	seed int64,
) (winRate, ciLow, ciHigh float64, significant bool) {
	if nRows <= 1 || iters <= 0 {
		return 0, 0, 0, false
	}
	rng := rand.New(rand.NewSource(seed))
	subLabels := make([]float64, nRows)
	subBase := make([]float64, nRows*nc)
	subCand := make([]float64, nRows*nc)
	diffs := make([]float64, 0, iters)
	wins := 0

	for b := 0; b < iters; b++ {
		for i := 0; i < nRows; i++ {
			pick := rng.Intn(nRows)
			subLabels[i] = labels[pick]
			if nc == 1 {
				subBase[i] = basePreds[pick]
				subCand[i] = candPreds[pick]
			} else {
				copy(subBase[i*nc:(i+1)*nc], basePreds[pick*nc:(pick+1)*nc])
				copy(subCand[i*nc:(i+1)*nc], candPreds[pick*nc:(pick+1)*nc])
			}
		}
		sBase, err1 := mt.Evaluate(subLabels, subBase)
		sCand, err2 := mt.Evaluate(subLabels, subCand)
		if err1 != nil || err2 != nil || math.IsNaN(sBase) || math.IsNaN(sCand) {
			continue
		}
		// diff > 0 恒表示 candidate 优于 baseline
		var diff float64
		if maximize {
			diff = sCand - sBase
		} else {
			diff = sBase - sCand
		}
		diffs = append(diffs, diff)
		if diff > 0 {
			wins++
		}
	}
	if len(diffs) == 0 {
		return 0, 0, 0, false
	}
	sort.Float64s(diffs)
	winRate = float64(wins) / float64(len(diffs))
	lowIdx := int(float64(len(diffs)-1) * 0.025)
	highIdx := int(float64(len(diffs)-1) * 0.975)
	ciLow = diffs[lowIdx]
	ciHigh = diffs[highIdx]
	significant = ciLow > 0
	return winRate, ciLow, ciHigh, significant
}
