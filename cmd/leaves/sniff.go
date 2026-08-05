package main

import (
	"flag"
	"fmt"
	"math"
	"strconv"

	"github.com/linkerlin/leaves/v2/data"
)

// sniffMaxScan 数据质量扫描行数上限，避免大文件 OOM/超时。
const sniffMaxScan = 5000

func cmdSniff(args []string) error {
	fs := flag.NewFlagSet("sniff", flag.ContinueOnError)
	fs.Usage = func() { fmt.Fprintln(fs.Output(), "用法: leaves sniff --data PATH [--metrics PATH]") }
	dataPath := fs.String("data", "", "数据路径（必需）")
	metricsPath := fs.String("metrics", "", "输出 JSON（空=stdout）")
	naPolicy := fs.String("na-policy", "error", "缺失值策略：error|skip-row（影响画像行数）")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *dataPath == "" {
		return errUsage("--data 必需")
	}

	sniff, err := data.SniffFileFormat(*dataPath)
	if err != nil {
		return errAgentWrap("data_load", fmt.Sprintf("sniff: %v", err),
			"检查文件路径与可读性", false, err)
	}
	dm, err := loadMatrix(*dataPath, *naPolicy)
	if err != nil {
		return err
	}

	// Matrix.NumCol() 始终是特征维：标签在 Labels()，排序 qid 在 Groups，均不计入列数。
	nRows, nFeat := dm.NumRow(), dm.NumCol()
	ls := labelStats(dm.Labels())
	isRank := sniff.Format == data.FormatRanking
	hasLabel := len(dm.Labels()) == nRows || isRank

	suggObj, suggMetric := suggestObjective(ls, sniff.Format)
	doc := map[string]any{
		"schema_version":      metricsSchemaVersion,
		"file":                *dataPath,
		"format":              formatName(sniff.Format),
		"n_rows":              nRows,
		"n_cols":              nFeat, // 与 Matrix 对齐：特征列数（非文件原始列数）
		"n_features":          nFeat,
		"has_label":           hasLabel,
		"has_qid":             isRank,
		"label":               ls,
		"suggested_objective": suggObj,
		"suggested_metric":    suggMetric,
		"data_quality":        scanDataQuality(dm, namesOf(dm)),
	}
	if names := featureNamesOf(dm); len(names) > 0 {
		doc["feature_names"] = names
	}
	if d, ok := dm.(*data.Dense); ok && d.SkippedRows > 0 {
		doc["skipped_rows"] = d.SkippedRows
		if dq, ok := doc["data_quality"].(map[string]any); ok {
			dq["skipped_rows"] = d.SkippedRows
			doc["data_quality"] = dq
		}
	}
	return writeJSON(*metricsPath, doc)
}

func namesOf(dm data.Matrix) []string {
	return featureNamesOf(dm)
}

// scanDataQuality 对已加载矩阵做轻量质量报告（不改数据；Agent 预知风险用）。
func scanDataQuality(dm data.Matrix, featureNames []string) map[string]any {
	n, c := dm.NumRow(), dm.NumCol()
	scanN := n
	if scanN > sniffMaxScan {
		scanN = sniffMaxScan
	}
	nanCells := 0
	infCells := 0
	// per-feature min/max 用于常数列检测
	minF := make([]float64, c)
	maxF := make([]float64, c)
	for j := 0; j < c; j++ {
		minF[j] = math.Inf(1)
		maxF[j] = math.Inf(-1)
	}
	buf := make([]float64, c)
	for i := 0; i < scanN; i++ {
		if err := dm.Row(i, buf); err != nil {
			continue
		}
		for j := 0; j < c; j++ {
			v := buf[j]
			if math.IsNaN(v) {
				nanCells++
				continue
			}
			if math.IsInf(v, 0) {
				infCells++
				continue
			}
			if v < minF[j] {
				minF[j] = v
			}
			if v > maxF[j] {
				maxF[j] = v
			}
		}
	}
	var constant []string
	for j := 0; j < c; j++ {
		if minF[j] == maxF[j] && !math.IsInf(minF[j], 0) {
			name := "f" + strconv.Itoa(j)
			if j < len(featureNames) && featureNames[j] != "" {
				name = featureNames[j]
			}
			constant = append(constant, name)
		}
	}
	warnings := []string{}
	if nanCells > 0 {
		warnings = append(warnings, "nan_cells="+strconv.Itoa(nanCells))
	}
	if infCells > 0 {
		warnings = append(warnings, "inf_cells="+strconv.Itoa(infCells))
	}
	if len(constant) > 0 {
		warnings = append(warnings, "constant_features="+strconv.Itoa(len(constant)))
	}
	// 二分类不均衡
	labels := dm.Labels()
	var imbalance any
	if len(labels) > 0 {
		pos, neg := 0, 0
		for _, y := range labels {
			if y == 1 {
				pos++
			} else if y == 0 {
				neg++
			}
		}
		if pos+neg == len(labels) && pos > 0 && neg > 0 {
			ratio := float64(pos) / float64(pos+neg)
			imbalance = ratio
			if ratio < 0.05 || ratio > 0.95 {
				warnings = append(warnings, fmt.Sprintf("label_imbalance_ratio=%.4f", ratio))
			}
		}
	}
	if n < 50 {
		warnings = append(warnings, "small_n_rows")
	}
	return map[string]any{
		"numeric":           true, // 能加载即视为数值矩阵
		"scanned_rows":      scanN,
		"total_rows":        n,
		"nan_cells":         nanCells,
		"inf_cells":         infCells,
		"constant_features": constant,
		"label_pos_ratio":   imbalance,
		"warnings":          warnings,
	}
}

// featureNamesOf 从 Dense / DenseWithGroups 取特征名（若有）。
func featureNamesOf(dm data.Matrix) []string {
	switch m := dm.(type) {
	case *data.Dense:
		return m.FNames
	case *data.DenseWithGroups:
		return m.FNames
	default:
		return nil
	}
}

func labelStats(labels []float64) map[string]any {
	if len(labels) == 0 {
		return map[string]any{"detected": false}
	}
	minV, maxV := labels[0], labels[0]
	uniq := map[float64]struct{}{}
	allInt := true
	for _, y := range labels {
		if y < minV {
			minV = y
		}
		if y > maxV {
			maxV = y
		}
		uniq[y] = struct{}{}
		if y != float64(int64(y)) {
			allInt = false
		}
	}
	kind := "regression"
	if allInt {
		switch {
		case len(uniq) == 2 && minV >= 0 && maxV <= 1:
			kind = "binary"
		case len(uniq) <= 20:
			kind = "classification"
		}
	}
	return map[string]any{
		"detected": true, "min": minV, "max": maxV,
		"n_unique": len(uniq), "kind": kind,
	}
}

func suggestObjective(ls map[string]any, format data.FileFormat) (string, string) {
	if format == data.FormatRanking {
		return "rank:ndcg", "ndcg@10"
	}
	kind, _ := ls["kind"].(string)
	switch kind {
	case "binary":
		return "binary:logistic", "logloss"
	case "classification":
		return "multi:softmax", "mlogloss"
	default:
		return "reg:squarederror", "rmse"
	}
}

func formatName(f data.FileFormat) string {
	switch f {
	case data.FormatCSV:
		return "csv"
	case data.FormatLIBSVM:
		return "libsvm"
	case data.FormatRanking:
		return "ranking"
	case data.FormatTSVLabelLast:
		return "tsv_label_last"
	default:
		return "auto"
	}
}
