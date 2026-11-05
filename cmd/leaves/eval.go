package main

import (
	"flag"
	"fmt"
	"strings"

	"github.com/linkerlin/leaves/v2/data"
	leavesio "github.com/linkerlin/leaves/v2/io"
	"github.com/linkerlin/leaves/v2/metrics"
)

func cmdEval(args []string) error {
	fs := flag.NewFlagSet("eval", flag.ContinueOnError)
	fs.Usage = func() { fmt.Fprintln(fs.Output(), "用法: leaves eval --model PATH --data PATH [flags]") }
	modelPath := fs.String("model", "", "模型路径（必需）")
	dataPath := fs.String("data", "", "数据路径（必需）")
	evalMetric := fs.String("eval-metric", "", "评估指标（默认 rmse）")
	objective := fs.String("objective", "", "目标函数（margin→pred 变换；可从 metric 推断）")
	metricsPath := fs.String("metrics", "", "输出 metrics.json（空=stdout）")
	naPolicy := fs.String("na-policy", "error", "缺失值策略：error|skip-row|nan")
	autoEncode := fs.Bool("auto-encode", false, "将非数值字符串特征列自动按字典序编码（默认优先复用模型内 category_maps）")
	numTarget := fs.Int("num-target", 0, "多目标评估：CSV 末 N 列标签（≥2；可从模型 n_output_groups 推断）")
	labelCol := fs.String("label-col", "", "显式指定标签列名或 0-based 索引（WP-22）")
	dropCols := fs.String("drop-cols", "", "逗号分隔需丢弃的列名或索引（如 ID、时间戳）")
	weightCol := fs.String("weight-col", "", "样本权重列名或 0-based 索引（WP-41）")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *modelPath == "" || *dataPath == "" {
		return errUsage("--model 与 --data 必需")
	}
	metric := *evalMetric
	obj := *objective
	if obj == "" {
		// 从模型元数据自动推断 objective（支持 leaves.json / XGB / LGB / sklearn / ONNX，WP-26）
		if _, irObj, _, err := leavesio.LoadModelIR(*modelPath); err == nil {
			obj = irObj
		}
	}
	if metric == "" {
		metric = defaultMetric(obj)
	}

	m, err := loadEnsemble(*modelPath)
	if err != nil {
		return fmt.Errorf("load model: %w", err)
	}
	defer m.Close()

	nc := m.NOutputGroups()
	if nc < 1 {
		nc = 1
	}
	// 多目标：优先 CLI --num-target；否则模型 groups>1 且非 multi 时自动用 groups
	nt := *numTarget
	if nt < 2 && nc > 1 && !isMulticlassObjective(obj) {
		nt = nc
	}

	catMaps := loadModelCategoryMaps(*modelPath)
	expectedFeats := loadModelFeatureNames(*modelPath)
	dm, err := loadMatrixFull(*dataPath, *naPolicy, nt, *labelCol, *dropCols, "", *weightCol, *autoEncode, catMaps, expectedFeats)
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

	preds := make([]float64, dm.NumRow()*nc)
	if err := m.PredictDense(vals, dm.NumRow(), dm.NumCol(), preds, 0, 0); err != nil {
		return fmt.Errorf("predict: %w", err)
	}

	groups := groupsOf(dm)
	mt, err := metrics.Resolve(metric, metrics.Options{NumClass: nc, Groups: groups})
	if err != nil {
		return fmt.Errorf("resolve metric %q: %w", metric, err)
	}
	metricPreds := metricInputs(metric, obj, preds, nc)
	labels := dm.Labels()
	if nt >= 2 {
		if mtm, ok := data.AsMultiTarget(dm); ok {
			labels = mtm.Targets()
			// 多目标回归：pred 已是 n*k 扁平 margin，与 Targets 对齐；不做 multiclass 变换
			if !isMulticlassObjective(obj) {
				metricPreds = preds
			}
		}
	}
	score, err := metrics.Evaluate(mt, labels, metricPreds, groups)
	if err != nil {
		return fmt.Errorf("evaluate: %w", err)
	}

	doc := metricsDoc{
		Objective: obj,
		Metric:    metric,
		Value:     score,
		Maximize:  mt.HigherIsBetter(),
		NRows:     dm.NumRow(),
		NFeatures: dm.NumCol(),
	}
	if obj == "binary:logistic" && nc == 1 {
		probs := make([]float64, len(preds))
		for i, raw := range preds {
			probs[i] = sigmoid(raw)
		}
		doc.BinaryDiagnostics = computeBinaryDiagnostics(labels, probs)
	} else if isMulticlassObjective(obj) && nc >= 2 {
		doc.MulticlassDiagnostics = computeMulticlassDiagnostics(labels, preds, nc)
	} else if nt < 2 && nc == 1 && (obj == "" || obj == "reg:squarederror" || obj == "reg:tweedie" || obj == "reg:gamma" || obj == "count:poisson") {
		if len(groups) > 0 && strings.HasPrefix(metric, "ndcg") {
			doc.RankingDiagnostics = computeRankingDiagnostics(labels, preds, groups)
		} else {
			regPreds := metricInputs("rmse", obj, preds, 1)
			doc.RegressionDiagnostics = computeRegressionDiagnostics(labels, regPreds)
		}
	} else if len(groups) > 0 {
		doc.RankingDiagnostics = computeRankingDiagnostics(labels, preds, groups)
	}
	if nt >= 2 {
		doc.Params = &paramsRecord{NumTarget: nt, EvalMetric: metric}
	}
	return writeMetrics(*metricsPath, doc)
}
