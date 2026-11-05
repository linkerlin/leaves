package main

import (
	"flag"
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"

	"github.com/linkerlin/leaves/v2/data"
)

// sniffMaxScan 数据质量扫描行数上限，避免大文件 OOM/超时。
const sniffMaxScan = 5000

func cmdSniff(args []string) error {
	fs := flag.NewFlagSet("sniff", flag.ContinueOnError)
	fs.Usage = func() { fmt.Fprintln(fs.Output(), "用法: leaves sniff --data PATH [--metrics PATH]") }
	dataPath := fs.String("data", "", "数据路径（必需）")
	metricsPath := fs.String("metrics", "", "输出 JSON（空=stdout）")
	naPolicy := fs.String("na-policy", "error", "缺失值策略：error|skip-row|nan（影响画像行数）")
	autoEncode := fs.Bool("auto-encode", false, "将非数值字符串特征列自动按字典序编码（WP-31）")
	labelCol := fs.String("label-col", "", "显式指定标签列名或 0-based 索引（WP-22）")
	dropCols := fs.String("drop-cols", "", "逗号分隔需丢弃的列名或索引（如 ID、时间戳）")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *dataPath == "" {
		return errUsage("--data 必需")
	}

	doc, err := buildSniffReport(*dataPath, *naPolicy, *labelCol, *dropCols, *autoEncode)
	if err != nil {
		return err
	}
	return writeJSON(*metricsPath, doc)
}

func buildSniffReport(dataPath, naPolicy, labelCol, dropCols string, autoEncode bool) (map[string]any, error) {
	sniff, err := data.SniffFileFormat(dataPath)
	if err != nil {
		return nil, errAgentWrap("data_load", fmt.Sprintf("sniff: %v", err),
			"检查文件路径与可读性", false, err)
	}
	dm, err := loadMatrixWithColsAndCats(dataPath, naPolicy, 0, labelCol, dropCols, "", autoEncode, nil)
	autoFallbackNumeric := true
	if err != nil {
		// 若用户未显式开启 --auto-encode 或 --na-policy nan，sniff 作为只读探测工具自动以宽容模式重试画像，
		// 以便向智能体报告 categorical_columns / nan_cells / id_like_columns，而非直接报错中断探测。
		dm2, err2 := loadMatrixWithColsAndCats(dataPath, "nan", 0, labelCol, dropCols, "", true, nil)
		if err2 != nil {
			return nil, err
		}
		dm = dm2
		autoFallbackNumeric = false
	}

	// Matrix.NumCol() 始终是特征维：标签在 Labels()，排序 qid 在 Groups，均不计入列数。
	nRows, nFeat := dm.NumRow(), dm.NumCol()
	isRank := sniff.Format == data.FormatRanking
	missingLabel := false
	if d, ok := dm.(*data.Dense); ok && d.MissingLabel && !isRank {
		missingLabel = true
	}
	hasLabel := (len(dm.Labels()) == nRows || isRank) && !missingLabel
	var ls map[string]any
	var suggObj, suggMetric string
	if missingLabel {
		ls = map[string]any{"detected": false}
	} else {
		ls = labelStats(dm.Labels())
		suggObj, suggMetric = suggestObjective(ls, sniff.Format)
	}
	featNames := namesOf(dm)
	dq := scanDataQuality(dm, featNames)
	if !autoFallbackNumeric {
		dq["numeric"] = false
	}
	if missingLabel {
		if w, ok := dq["warnings"].([]string); ok {
			dq["warnings"] = append(w, "missing_label")
		}
	}

	var catCols []string
	var idLikeCols []string
	var suggestedDrops []string
	dropSeen := map[string]bool{}
	addSuggestedDrop := func(c string) {
		c = strings.TrimSpace(c)
		if c == "" || dropSeen[c] {
			return
		}
		dropSeen[c] = true
		suggestedDrops = append(suggestedDrops, c)
	}

	if d, ok := dm.(*data.Dense); ok {
		for colName := range d.CategoryMaps {
			catCols = append(catCols, colName)
		}
		sort.Strings(catCols)
		for _, nm := range featNames {
			lower := strings.ToLower(strings.TrimSpace(nm))
			isIDName := lower == "id" || lower == "uuid" || lower == "guid" ||
				strings.HasSuffix(lower, "_id") || strings.HasSuffix(lower, "id") && (strings.HasPrefix(lower, "user") || strings.HasPrefix(lower, "item") || strings.HasPrefix(lower, "row") || strings.HasPrefix(lower, "sample"))
			if isIDName {
				idLikeCols = append(idLikeCols, nm)
				addSuggestedDrop(nm)
				continue
			}
			if m, isCat := d.CategoryMaps[nm]; isCat && nRows >= 10 && len(m) == nRows {
				idLikeCols = append(idLikeCols, nm)
				addSuggestedDrop(nm)
			}
		}
	}
	if cf, ok := dq["constant_features"].([]string); ok {
		for _, c := range cf {
			addSuggestedDrop(c)
		}
	}
	if len(catCols) > 0 {
		if w, ok := dq["warnings"].([]string); ok {
			dq["warnings"] = append(w, fmt.Sprintf("categorical_columns=%d (use --auto-encode)", len(catCols)))
		}
	}
	if len(idLikeCols) > 0 {
		if w, ok := dq["warnings"].([]string); ok {
			dq["warnings"] = append(w, fmt.Sprintf("id_like_columns=%d", len(idLikeCols)))
		}
	}

	doc := map[string]any{
		"schema_version":      metricsSchemaVersion,
		"file":                dataPath,
		"format":              formatName(sniff.Format),
		"n_rows":              nRows,
		"n_cols":              nFeat, // 与 Matrix 对齐：特征列数（非文件原始列数）
		"n_features":          nFeat,
		"has_label":           hasLabel,
		"has_qid":             isRank,
		"label":               ls,
		"suggested_objective": suggObj,
		"suggested_metric":    suggMetric,
		"data_quality":        dq,
	}
	if len(catCols) > 0 {
		doc["categorical_columns"] = catCols
	}
	if len(idLikeCols) > 0 {
		doc["id_like_columns"] = idLikeCols
	}
	if len(suggestedDrops) > 0 {
		doc["suggested_drop_cols"] = suggestedDrops
	}
	if d, ok := dm.(*data.Dense); ok {
		if d.LabelName != "" {
			doc["label_col"] = d.LabelName
		}
		if len(d.DroppedCols) > 0 {
			doc["dropped_cols"] = d.DroppedCols
		}
		if len(d.CategoryMaps) > 0 && (autoEncode || !autoFallbackNumeric) {
			doc["category_maps"] = d.CategoryMaps
		}
	}
	if len(featNames) > 0 {
		doc["feature_names"] = featNames
	}
	if d, ok := dm.(*data.Dense); ok && d.SkippedRows > 0 {
		doc["skipped_rows"] = d.SkippedRows
		if dq, ok := doc["data_quality"].(map[string]any); ok {
			dq["skipped_rows"] = d.SkippedRows
			doc["data_quality"] = dq
		}
	}
	return doc, nil
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
