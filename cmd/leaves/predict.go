package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/linkerlin/leaves/v2/data"
	"github.com/linkerlin/leaves/v2/predict"
)

func cmdPredict(args []string) error {
	fs := flag.NewFlagSet("predict", flag.ContinueOnError)
	fs.Usage = func() {
		fmt.Fprintln(fs.Output(), "用法: leaves predict --model PATH --data PATH --out PATH [flags]")
	}
	modelPath := fs.String("model", "", "模型路径（必需）")
	dataPath := fs.String("data", "", "数据路径（必需）")
	out := fs.String("out", "", "输出路径（必需）")
	metricsPath := fs.String("metrics", "", "可选输出模型推理统计摘要 JSON 路径（WP-45）")
	objective := fs.String("objective", "", "目标函数；空=从模型推断")
	format := fs.String("format", "jsonl", "jsonl|csv（默认 jsonl；csv 出单 prediction 列）")
	outputKind := fs.String("output", "value", "预测输出类型：value（默认）| margin | leaf | contrib（WP-27）")
	idCol := fs.String("id-col", "", "样本 ID 列名或 0-based 索引（跳过特征并在输出中透传 id，WP-27）")
	labelCol := fs.String("label-col", "", "显式指定标签列名或 0-based 索引（预测时自动跳过，WP-22）")
	dropCols := fs.String("drop-cols", "", "逗号分隔需丢弃的列名或索引")
	naPolicy := fs.String("na-policy", "error", "缺失值策略：error|skip-row|nan")
	autoEncode := fs.Bool("auto-encode", false, "将非数值字符串特征列自动按字典序编码（默认优先复用模型内 category_maps）")
	threshold := fs.Float64("threshold", 0.5, "二分类（binary:logistic）分类阈值（默认 0.5；输出 predicted_label，WP-32）")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *modelPath == "" || *dataPath == "" || *out == "" {
		return errUsage("--model/--data/--out 必需")
	}

	t0 := time.Now()
	obj := *objective
	if obj == "" {
		if _, irObj, _, _, err := loadModelIRWithQuant(*modelPath); err == nil {
			obj = irObj
		}
	}

	m, err := loadEnsemble(*modelPath)
	if err != nil {
		return fmt.Errorf("load model: %w", err)
	}
	defer m.Close()

	catMaps := loadModelCategoryMaps(*modelPath)
	expectedFeats := loadModelFeatureNames(*modelPath)
	dm, err := loadMatrixFull(*dataPath, *naPolicy, 0, *labelCol, *dropCols, *idCol, "", *autoEncode, catMaps, expectedFeats)
	if err != nil {
		return err
	}
	vals, err := denseVals(dm)
	if err != nil {
		return err
	}

	var idVals []string
	var idName string
	if d, ok := dm.(*data.Dense); ok && len(d.IDValues) == dm.NumRow() {
		idVals = d.IDValues
		idName = d.IDName
	}
	if idName == "" {
		idName = "id"
	}

	_ = os.MkdirAll(filepath.Dir(*out), 0o755)
	f, err := os.Create(*out)
	if err != nil {
		return err
	}
	defer f.Close()

	n := dm.NumRow()
	cols := dm.NumCol()
	outMode := strings.ToLower(strings.TrimSpace(*outputKind))
	if outMode == "" {
		outMode = "value"
	}

	writePredictMetrics := func(extra map[string]any) error {
		if strings.TrimSpace(*metricsPath) == "" {
			return nil
		}
		summary := map[string]any{
			"schema_version": metricsSchemaVersion,
			"model":          *modelPath,
			"data":           *dataPath,
			"out":            *out,
			"objective":      obj,
			"output_kind":    outMode,
			"format":         *format,
			"n_rows":         n,
			"n_features":     cols,
			"elapsed_ms":     time.Since(t0).Milliseconds(),
		}
		for k, v := range extra {
			summary[k] = v
		}
		return writeJSON(*metricsPath, summary)
	}

	switch outMode {
	case "leaf", "leaves":
		nTrees := len(m.NLeaves())
		if nTrees <= 0 {
			nTrees = m.NEstimators() * m.NRawOutputGroups()
		}
		if nTrees <= 0 {
			return fmt.Errorf("predict leaf requires tree model")
		}
		leafOut := make([]float64, n*nTrees)
		req := predict.Request{
			Matrix: predict.DenseMatrix{Values: vals, Rows: n, Cols: cols},
			Output: predict.OutputLeaf,
		}
		if err := m.PredictWithRequest(req, leafOut); err != nil {
			return fmt.Errorf("predict leaf: %w", err)
		}
		if *format == "csv" {
			if err := writeCSVMatrix(f, "leaf", n, nTrees, leafOut, idName, idVals, true); err != nil {
				return err
			}
			return writePredictMetrics(map[string]any{"n_trees": nTrees})
		}
		enc := json.NewEncoder(f)
		for i := 0; i < n; i++ {
			leaves := make([]int, nTrees)
			for t := 0; t < nTrees; t++ {
				leaves[t] = int(leafOut[i*nTrees+t])
			}
			rec := map[string]any{"row": i, "leaves": leaves}
			if i < len(idVals) {
				rec["id"] = idVals[i]
			}
			if err := enc.Encode(rec); err != nil {
				return err
			}
		}
		return writePredictMetrics(map[string]any{"n_trees": nTrees})

	case "contrib", "contribution", "shap":
		nGroups := m.NRawOutputGroups()
		if nGroups <= 0 {
			nGroups = 1
		}
		stride := (cols + 1) * nGroups
		contribOut := make([]float64, n*stride)
		req := predict.Request{
			Matrix: predict.DenseMatrix{Values: vals, Rows: n, Cols: cols},
			Output: predict.OutputContribution,
		}
		if err := m.PredictWithRequest(req, contribOut); err != nil {
			return fmt.Errorf("predict contrib: %w", err)
		}
		if *format == "csv" {
			if err := writeCSVMatrix(f, "contrib", n, stride, contribOut, idName, idVals, false); err != nil {
				return err
			}
			return writePredictMetrics(map[string]any{"stride": stride})
		}
		enc := json.NewEncoder(f)
		for i := 0; i < n; i++ {
			row := contribOut[i*stride : (i+1)*stride]
			rec := map[string]any{"row": i}
			if i < len(idVals) {
				rec["id"] = idVals[i]
			}
			if nGroups == 1 {
				c := make([]float64, cols)
				copy(c, row[:cols])
				rec["contributions"] = c
				rec["bias"] = row[cols]
			} else {
				all := make([]float64, stride)
				copy(all, row)
				rec["contributions"] = all
			}
			if err := enc.Encode(rec); err != nil {
				return err
			}
		}
		return writePredictMetrics(map[string]any{"stride": stride})

	case "value", "margin":
		// 继续下方标准 margin / value 输出
	default:
		return errUsage(fmt.Sprintf("未知 --output %q（支持 value|margin|leaf|contrib）", *outputKind))
	}

	nc := m.NOutputGroups()
	if nc < 1 {
		nc = 1
	}
	preds := make([]float64, n*nc)
	if err := m.PredictDense(vals, n, cols, preds, 0, 0); err != nil {
		return fmt.Errorf("predict: %w", err)
	}

	multiClass := isMulticlassObjective(obj) && outMode != "margin"
	// 对单输出模型计算变换后的 prediction 值及统计摘要（WP-45）
	var transformed []float64
	extraStats := map[string]any{}
	if nc == 1 {
		transformed = make([]float64, n)
		predMin := math.Inf(1)
		predMax := math.Inf(-1)
		predSum := 0.0
		posCount := 0
		for i := 0; i < n; i++ {
			v := applyObjectiveLink(preds[i], obj, outMode)
			transformed[i] = v
			if v < predMin {
				predMin = v
			}
			if v > predMax {
				predMax = v
			}
			predSum += v
			if obj == "binary:logistic" && outMode != "margin" && v >= *threshold {
				posCount++
			}
		}
		if n > 0 {
			predMean := predSum / float64(n)
			var sqDiff float64
			for _, v := range transformed {
				d := v - predMean
				sqDiff += d * d
			}
			extraStats["pred_min"] = round4(predMin)
			extraStats["pred_max"] = round4(predMax)
			extraStats["pred_mean"] = round4(predMean)
			extraStats["pred_std"] = round4(math.Sqrt(sqDiff / float64(n)))
			if obj == "binary:logistic" && outMode != "margin" {
				extraStats["threshold"] = *threshold
				extraStats["positive_count"] = posCount
				extraStats["positive_rate"] = round4(float64(posCount) / float64(n))
			}
		}
	} else if multiClass && n > 0 {
		classCounts := make(map[string]int)
		for i := 0; i < n; i++ {
			c := argmax(preds[i*nc : (i+1)*nc])
			classCounts[fmt.Sprintf("%d", c)]++
		}
		extraStats["class_counts"] = classCounts
	}

	if *format == "csv" {
		csvPreds := preds
		if nc == 1 && len(transformed) == n {
			csvPreds = transformed
		}
		if err := writeCSVDenseObjWithID(f, nc, n, csvPreds, multiClass, idName, idVals); err != nil {
			return err
		}
		return writePredictMetrics(extraStats)
	}

	enc := json.NewEncoder(f)
	for i := 0; i < n; i++ {
		row := preds[i*nc : (i+1)*nc]
		rec := map[string]any{"row": i}
		if i < len(idVals) {
			rec["id"] = idVals[i]
		}
		if nc == 1 {
			rec["margin"] = row[0]
			if outMode != "margin" {
				val := transformed[i]
				rec["prediction"] = val
				if obj == "binary:logistic" {
					rec["probability"] = val
					predLabel := 0
					if val >= *threshold {
						predLabel = 1
					}
					rec["predicted_label"] = predLabel
				}
			}
		} else if multiClass {
			rec["class"] = argmax(row)
			rec["probabilities"] = softmaxRows(row, nc)
		} else {
			margins := make([]float64, nc)
			copy(margins, row)
			rec["margins"] = margins
			rec["predictions"] = margins
		}
		if err := enc.Encode(rec); err != nil {
			return err
		}
	}
	return writePredictMetrics(extraStats)
}

func isMulticlassObjective(obj string) bool {
	return obj == "multi:softmax" || obj == "multi:softprob"
}

func isLogLinkObjective(obj string) bool {
	switch obj {
	case "count:poisson", "reg:gamma", "reg:tweedie", "survival:cox", "survival:aft":
		return true
	default:
		return false
	}
}

func applyObjectiveLink(margin float64, obj, outMode string) float64 {
	if outMode == "margin" {
		return margin
	}
	if obj == "binary:logistic" {
		return sigmoid(margin)
	}
	if isLogLinkObjective(obj) {
		return math.Exp(margin)
	}
	return margin
}

func writeCSVDenseObjWithID(w io.Writer, numOut, n int, preds []float64, multiClass bool, idName string, idVals []string) error {
	hasID := len(idVals) == n
	idPrefix := func(i int) string {
		if !hasID {
			return ""
		}
		return idVals[i] + ","
	}
	headerPrefix := ""
	if hasID {
		headerPrefix = idName + ","
	}
	if numOut > 1 && multiClass {
		fmt.Fprintln(w, headerPrefix+"prediction")
		for i := 0; i < n; i++ {
			fmt.Fprintf(w, "%s%d\n", idPrefix(i), argmax(preds[i*numOut:(i+1)*numOut]))
		}
		return nil
	}
	if numOut > 1 {
		fmt.Fprint(w, headerPrefix)
		for k := 0; k < numOut; k++ {
			if k > 0 {
				fmt.Fprint(w, ",")
			}
			fmt.Fprintf(w, "pred_%d", k)
		}
		fmt.Fprintln(w)
		for i := 0; i < n; i++ {
			fmt.Fprint(w, idPrefix(i))
			for k := 0; k < numOut; k++ {
				if k > 0 {
					fmt.Fprint(w, ",")
				}
				fmt.Fprintf(w, "%g", preds[i*numOut+k])
			}
			fmt.Fprintln(w)
		}
		return nil
	}
	fmt.Fprintln(w, headerPrefix+"prediction")
	for i := 0; i < n; i++ {
		fmt.Fprintf(w, "%s%g\n", idPrefix(i), preds[i])
	}
	return nil
}

func writeCSVMatrix(w io.Writer, colPrefix string, n, stride int, vals []float64, idName string, idVals []string, asInt bool) error {
	hasID := len(idVals) == n
	if hasID {
		fmt.Fprint(w, idName+",")
	}
	for k := 0; k < stride; k++ {
		if k > 0 {
			fmt.Fprint(w, ",")
		}
		fmt.Fprintf(w, "%s_%d", colPrefix, k)
	}
	fmt.Fprintln(w)
	for i := 0; i < n; i++ {
		if hasID {
			fmt.Fprintf(w, "%s,", idVals[i])
		}
		for k := 0; k < stride; k++ {
			if k > 0 {
				fmt.Fprint(w, ",")
			}
			v := vals[i*stride+k]
			if asInt {
				fmt.Fprintf(w, "%d", int(v))
			} else {
				fmt.Fprintf(w, "%g", v)
			}
		}
		fmt.Fprintln(w)
	}
	return nil
}
