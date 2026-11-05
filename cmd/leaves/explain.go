package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/linkerlin/leaves/v2/explain"
	leavesio "github.com/linkerlin/leaves/v2/io"
	"github.com/linkerlin/leaves/v2/model"
)

func cmdExplain(args []string) error {
	fs := flag.NewFlagSet("explain", flag.ContinueOnError)
	fs.Usage = func() {
		fmt.Fprintln(fs.Output(), "用法: leaves explain --model PATH [--type importance|shap|shap-summary|interaction|dump] [--data PATH] [--max-rows N] [--metrics PATH]")
	}
	modelPath := fs.String("model", "", "模型路径（支持 leaves.json/XGB/LGB/sklearn/ONNX，必需）")
	kind := fs.String("type", "importance", "importance（无需数据）| shap | shap-summary | interaction | dump")
	impType := fs.String("importance-type", "gain", "importance 子类型：gain（默认）| weight | cover | total_gain | total_cover（WP-37）")
	summary := fs.Bool("summary", false, "--type shap 时输出全局 mean_abs_shap 排序汇总而非逐行数组（WP-37）")
	dumpFormat := fs.String("dump-format", "json", "--type dump 时的格式：json（默认）| text | dot（WP-37）")
	dataPath := fs.String("data", "", "数据路径（--type shap/interaction 必需）")
	maxRows := fs.Int("max-rows", 100, "SHAP/interaction 最多样本数（默认 100，上限 500）")
	metricsPath := fs.String("metrics", "", "输出路径（空=stdout）")
	labelCol := fs.String("label-col", "", "显式指定标签列名或 0-based 索引（WP-22）")
	dropCols := fs.String("drop-cols", "", "逗号分隔需丢弃的列名或索引")
	naPolicy := fs.String("na-policy", "error", "缺失值策略：error|skip-row|nan")
	autoEncode := fs.Bool("auto-encode", false, "将非数值字符串特征列自动按字典序编码（默认优先复用模型内 category_maps）")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *modelPath == "" {
		return errUsage("--model 必需")
	}
	ir, _, _, err := leavesio.LoadModelIR(*modelPath)
	if err != nil {
		return fmt.Errorf("parse model: %w", err)
	}
	if ir.Forest == nil {
		return fmt.Errorf("model has no forest（仅树模型支持 explain）")
	}

	switch strings.ToLower(strings.TrimSpace(*kind)) {
	case "importance":
		return explainImportance(ir, *impType, *dataPath, *naPolicy, *labelCol, *dropCols, *autoEncode, *metricsPath)
	case "shap":
		return explainSHAP(ir, *dataPath, *naPolicy, *labelCol, *dropCols, *autoEncode, *maxRows, *summary, *metricsPath)
	case "shap-summary", "shap_summary":
		return explainSHAP(ir, *dataPath, *naPolicy, *labelCol, *dropCols, *autoEncode, *maxRows, true, *metricsPath)
	case "interaction", "interactions":
		return explainInteraction(ir, *dataPath, *naPolicy, *labelCol, *dropCols, *autoEncode, *maxRows, *metricsPath)
	case "dump":
		return explainDump(ir, *dumpFormat, *dataPath, *naPolicy, *labelCol, *dropCols, *autoEncode, *metricsPath)
	default:
		return errUsage("--type 仅支持 importance|shap|shap-summary|interaction|dump")
	}
}

func resolveExplainFeatureNames(ir *model.ModelIR, dataPath, naPolicy, labelCol, dropCols string, autoEncode bool) []string {
	names := ir.FeatureNames
	if len(names) == 0 && dataPath != "" {
		if dm, err := loadMatrixWithColsAndCats(dataPath, naPolicy, 0, labelCol, dropCols, "", autoEncode, ir.CategoryMaps); err == nil {
			names = featureNamesOf(dm)
		}
	}
	return names
}

func explainImportance(ir *model.ModelIR, impTypeStr, dataPath, naPolicy, labelCol, dropCols string, autoEncode bool, metricsPath string) error {
	names := resolveExplainFeatureNames(ir, dataPath, naPolicy, labelCol, dropCols, autoEncode)
	var primaryKind explain.ImportanceType
	normType := strings.ToLower(strings.TrimSpace(impTypeStr))
	switch normType {
	case "", "gain":
		primaryKind = explain.ImportanceGain
		normType = "gain"
	case "weight", "split", "splits":
		primaryKind = explain.ImportanceWeight
		normType = "weight"
	case "cover":
		primaryKind = explain.ImportanceCover
		normType = "cover"
	case "total_gain", "gain_pct":
		primaryKind = explain.ImportanceTotalGain
		normType = "total_gain"
	case "total_cover", "cover_pct":
		primaryKind = explain.ImportanceTotalCover
		normType = "total_cover"
	default:
		return errUsage(fmt.Sprintf("未知 --importance-type %q（支持 gain|weight|cover|total_gain|total_cover）", impTypeStr))
	}

	fiPrimary := explain.ComputeImportance(ir.Forest, primaryKind, names)
	fiGain := explain.ComputeImportance(ir.Forest, explain.ImportanceGain, names)
	fiWeight := explain.ComputeImportance(ir.Forest, explain.ImportanceWeight, names)
	fiCover := explain.ComputeImportance(ir.Forest, explain.ImportanceCover, names)
	fiGainPct := explain.ComputeImportance(ir.Forest, explain.ImportanceTotalGain, names)
	fiCoverPct := explain.ComputeImportance(ir.Forest, explain.ImportanceTotalCover, names)

	feats := []map[string]any{}
	ranked := []map[string]any{}
	var zeroGain []string
	nFeat := 0
	if fiPrimary != nil {
		nFeat = len(fiPrimary.Scores)
		for i := 0; i < nFeat; i++ {
			nm := fiPrimary.Names[i]
			score := fiPrimary.Scores[i]
			gain := 0.0
			if fiGain != nil && i < len(fiGain.Scores) {
				gain = fiGain.Scores[i]
			}
			weight := 0.0
			if fiWeight != nil && i < len(fiWeight.Scores) {
				weight = fiWeight.Scores[i]
			}
			cover := 0.0
			if fiCover != nil && i < len(fiCover.Scores) {
				cover = fiCover.Scores[i]
			}
			gainPct := 0.0
			if fiGainPct != nil && i < len(fiGainPct.Scores) {
				gainPct = fiGainPct.Scores[i]
			}
			coverPct := 0.0
			if fiCoverPct != nil && i < len(fiCoverPct.Scores) {
				coverPct = fiCoverPct.Scores[i]
			}
			if gain <= 0 {
				zeroGain = append(zeroGain, nm)
			}
			entry := map[string]any{
				"name":        nm,
				"score":       score,
				"gain":        gain,
				"split_count": int(math.Round(weight)),
				"cover":       cover,
				"gain_pct":    gainPct,
				"cover_pct":   coverPct,
			}
			feats = append(feats, entry)
			ranked = append(ranked, entry)
		}
		sort.SliceStable(ranked, func(i, j int) bool {
			si, _ := ranked[i]["score"].(float64)
			sj, _ := ranked[j]["score"].(float64)
			return si > sj
		})
	}
	return writeJSON(metricsPath, map[string]any{
		"schema_version":     metricsSchemaVersion,
		"type":               normType,
		"n_features":         nFeat,
		"features":           feats,
		"ranked_features":    ranked,
		"zero_gain_features": zeroGain,
	})
}

func explainSHAP(ir *model.ModelIR, dataPath, naPolicy, labelCol, dropCols string, autoEncode bool, maxRows int, summary bool, metricsPath string) error {
	if dataPath == "" {
		return errUsage("--type shap 需要 --data")
	}
	dm, err := loadMatrixFull(dataPath, naPolicy, 0, labelCol, dropCols, "", "", autoEncode, ir.CategoryMaps, ir.FeatureNames)
	if err != nil {
		return err
	}
	vals, err := denseVals(dm)
	if err != nil {
		return err
	}
	cols := dm.NumCol()
	n := len(vals) / cols
	if n > 500 {
		n = 500 // ponytail: SHAP 成本高，硬上限。
	}
	if maxRows > 0 && maxRows < n {
		n = maxRows
	}
	rows := make([][]float64, n)
	for i := 0; i < n; i++ {
		rows[i] = vals[i*cols : (i+1)*cols]
	}

	exp := explain.NewTreeExplainer(ir.Forest)
	names := ir.FeatureNames
	if len(names) == 0 {
		names = featureNamesOf(dm)
	}
	featureNameAt := func(j int) string {
		if j < len(names) && names[j] != "" {
			return names[j]
		}
		return fmt.Sprintf("f%d", j)
	}

	if ir.NOutputGroups > 1 {
		shapMulti, err := exp.ShapleyValuesMulticlass(rows)
		if err != nil {
			return fmt.Errorf("shap multiclass: %w", err)
		}
		bases := exp.ExpectedValues()
		if summary {
			g := ir.NOutputGroups
			meanAbsGlobal := make([]float64, cols)
			perClass := make([]map[string]any, g)
			for cIdx := 0; cIdx < g; cIdx++ {
				meanAbsC := make([]float64, cols)
				meanC := make([]float64, cols)
				for i := 0; i < n; i++ {
					for j := 0; j < cols; j++ {
						v := shapMulti[i][cIdx][j]
						meanAbsC[j] += math.Abs(v) / float64(n)
						meanC[j] += v / float64(n)
						meanAbsGlobal[j] += math.Abs(v) / float64(n*g)
					}
				}
				cFeats := make([]map[string]any, cols)
				for j := 0; j < cols; j++ {
					cFeats[j] = map[string]any{
						"name":          featureNameAt(j),
						"mean_abs_shap": meanAbsC[j],
						"mean_shap":     meanC[j],
					}
				}
				sort.SliceStable(cFeats, func(i, j int) bool {
					return cFeats[i]["mean_abs_shap"].(float64) > cFeats[j]["mean_abs_shap"].(float64)
				})
				bVal := 0.0
				if cIdx < len(bases) {
					bVal = bases[cIdx]
				}
				perClass[cIdx] = map[string]any{
					"class":    cIdx,
					"base":     bVal,
					"features": cFeats,
				}
			}
			globalFeats := make([]map[string]any, cols)
			var zeroShap []string
			for j := 0; j < cols; j++ {
				nm := featureNameAt(j)
				if meanAbsGlobal[j] <= 0 {
					zeroShap = append(zeroShap, nm)
				}
				globalFeats[j] = map[string]any{
					"name":          nm,
					"mean_abs_shap": meanAbsGlobal[j],
				}
			}
			sort.SliceStable(globalFeats, func(i, j int) bool {
				return globalFeats[i]["mean_abs_shap"].(float64) > globalFeats[j]["mean_abs_shap"].(float64)
			})
			return writeJSON(metricsPath, map[string]any{
				"schema_version":     metricsSchemaVersion,
				"type":               "shap_summary_multiclass",
				"n_rows":             n,
				"n_features":         cols,
				"n_classes":          g,
				"base_values":        bases,
				"features":           globalFeats,
				"per_class":          perClass,
				"zero_shap_features": zeroShap,
			})
		}

		out := make([]map[string]any, len(rows))
		for i := range rows {
			classes := make([]map[string]any, len(shapMulti[i]))
			for cIdx, classVals := range shapMulti[i] {
				bVal := 0.0
				if cIdx < len(bases) {
					bVal = bases[cIdx]
				}
				feats := make([]map[string]any, len(classVals))
				for j, v := range classVals {
					feats[j] = map[string]any{"name": featureNameAt(j), "value": v}
				}
				classes[cIdx] = map[string]any{
					"class":    cIdx,
					"base":     bVal,
					"features": feats,
				}
			}
			out[i] = map[string]any{
				"row":     i,
				"bases":   bases,
				"classes": classes,
			}
		}
		return writeJSON(metricsPath, out)
	}

	shapRows, err := exp.ShapleyValues(rows)
	if err != nil {
		return fmt.Errorf("shap: %w", err)
	}
	base := exp.ExpectedValue()

	if summary {
		meanAbs := make([]float64, cols)
		meanSigned := make([]float64, cols)
		for i := 0; i < n; i++ {
			for j := 0; j < cols; j++ {
				v := shapRows[i][j]
				meanAbs[j] += math.Abs(v) / float64(n)
				meanSigned[j] += v / float64(n)
			}
		}
		feats := make([]map[string]any, cols)
		var zeroShap []string
		for j := 0; j < cols; j++ {
			nm := featureNameAt(j)
			if meanAbs[j] <= 0 {
				zeroShap = append(zeroShap, nm)
			}
			feats[j] = map[string]any{
				"name":          nm,
				"mean_abs_shap": meanAbs[j],
				"mean_shap":     meanSigned[j],
			}
		}
		sort.SliceStable(feats, func(i, j int) bool {
			return feats[i]["mean_abs_shap"].(float64) > feats[j]["mean_abs_shap"].(float64)
		})
		return writeJSON(metricsPath, map[string]any{
			"schema_version":     metricsSchemaVersion,
			"type":               "shap_summary",
			"n_rows":             n,
			"n_features":         cols,
			"base_value":         base,
			"features":           feats,
			"zero_shap_features": zeroShap,
		})
	}

	out := make([]map[string]any, len(rows))
	for i := range rows {
		row := map[string]any{"row": i, "base": base}
		feats := []map[string]any{}
		for j := range shapRows[i] {
			feats = append(feats, map[string]any{"name": featureNameAt(j), "value": shapRows[i][j]})
		}
		row["features"] = feats
		out[i] = row
	}
	return writeJSON(metricsPath, out)
}

func explainInteraction(ir *model.ModelIR, dataPath, naPolicy, labelCol, dropCols string, autoEncode bool, maxRows int, metricsPath string) error {
	if dataPath == "" {
		return errUsage("--type interaction 需要 --data")
	}
	dm, err := loadMatrixFull(dataPath, naPolicy, 0, labelCol, dropCols, "", "", autoEncode, ir.CategoryMaps, ir.FeatureNames)
	if err != nil {
		return err
	}
	vals, err := denseVals(dm)
	if err != nil {
		return err
	}
	cols := dm.NumCol()
	n := len(vals) / cols
	if n > 200 {
		n = 200
	}
	if maxRows > 0 && maxRows < n {
		n = maxRows
	}
	rows := make([][]float64, n)
	for i := 0; i < n; i++ {
		rows[i] = vals[i*cols : (i+1)*cols]
	}

	exp := explain.NewTreeExplainer(ir.Forest)
	names := ir.FeatureNames
	if len(names) == 0 {
		names = featureNamesOf(dm)
	}
	featureNameAt := func(j int) string {
		if j < len(names) && names[j] != "" {
			return names[j]
		}
		return fmt.Sprintf("f%d", j)
	}

	meanAbsMat := make([][]float64, cols)
	for j := 0; j < cols; j++ {
		meanAbsMat[j] = make([]float64, cols)
	}

	if ir.NOutputGroups > 1 {
		interMulti, err := exp.InteractionValuesMulticlass(rows)
		if err != nil {
			return fmt.Errorf("interaction multiclass: %w", err)
		}
		g := ir.NOutputGroups
		denom := float64(n * g)
		for i := 0; i < n; i++ {
			for c := 0; c < g; c++ {
				for j := 0; j < cols; j++ {
					for k := 0; k < cols; k++ {
						meanAbsMat[j][k] += math.Abs(interMulti[i][c][j][k]) / denom
					}
				}
			}
		}
	} else {
		inter, err := exp.InteractionValues(rows)
		if err != nil {
			return fmt.Errorf("interaction: %w", err)
		}
		denom := float64(n)
		for i := 0; i < n; i++ {
			for j := 0; j < cols; j++ {
				for k := 0; k < cols; k++ {
					meanAbsMat[j][k] += math.Abs(inter[i][j][k]) / denom
				}
			}
		}
	}

	var pairs []map[string]any
	for j := 0; j < cols; j++ {
		for k := j + 1; k < cols; k++ {
			strength := meanAbsMat[j][k] + meanAbsMat[k][j]
			pairs = append(pairs, map[string]any{
				"feature_a":            featureNameAt(j),
				"feature_b":            featureNameAt(k),
				"mean_abs_interaction": strength,
			})
		}
	}
	sort.SliceStable(pairs, func(i, j int) bool {
		return pairs[i]["mean_abs_interaction"].(float64) > pairs[j]["mean_abs_interaction"].(float64)
	})
	if len(pairs) > 20 {
		pairs = pairs[:20]
	}

	featNamesOut := make([]string, cols)
	for j := 0; j < cols; j++ {
		featNamesOut[j] = featureNameAt(j)
	}

	return writeJSON(metricsPath, map[string]any{
		"schema_version":   metricsSchemaVersion,
		"type":             "interaction",
		"n_rows":           n,
		"n_features":       cols,
		"feature_names":    featNamesOut,
		"top_interactions": pairs,
		"mean_abs_matrix":  meanAbsMat,
	})
}

func explainDump(ir *model.ModelIR, dumpFormat, dataPath, naPolicy, labelCol, dropCols string, autoEncode bool, metricsPath string) error {
	names := resolveExplainFeatureNames(ir, dataPath, naPolicy, labelCol, dropCols, autoEncode)
	fmtMode := strings.ToLower(strings.TrimSpace(dumpFormat))
	switch fmtMode {
	case "", "json":
		b, err := explain.DumpJSON(ir.Forest, names)
		if err != nil {
			return err
		}
		var parsed any
		if err := json.Unmarshal(b, &parsed); err != nil {
			return err
		}
		return writeJSON(metricsPath, parsed)
	case "text", "txt":
		txt := explain.DumpText(ir.Forest, names)
		return writeRawTextOrStdout(metricsPath, txt)
	case "dot", "graphviz":
		dot := explain.DumpDOT(ir.Forest, names)
		return writeRawTextOrStdout(metricsPath, dot)
	default:
		return errUsage(fmt.Sprintf("未知 --dump-format %q（支持 json|text|dot）", dumpFormat))
	}
}

func writeRawTextOrStdout(path, content string) error {
	if !strings.HasSuffix(content, "\n") {
		content += "\n"
	}
	if path == "" || path == "-" {
		_, err := os.Stdout.WriteString(content)
		return err
	}
	_ = os.MkdirAll(filepath.Dir(path), 0o755)
	return os.WriteFile(path, []byte(content), 0o644)
}
