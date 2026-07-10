package main

import (
	"flag"
	"fmt"

	"github.com/linkerlin/leaves/data"
	"github.com/linkerlin/leaves/explain"
	leavesio "github.com/linkerlin/leaves/io"
	"github.com/linkerlin/leaves/model"
)

func cmdExplain(args []string) error {
	fs := flag.NewFlagSet("explain", flag.ContinueOnError)
	fs.Usage = func() {
		fmt.Fprintln(fs.Output(), "用法: leaves explain --model PATH [--type importance|shap] [--data PATH] [--max-rows N] [--metrics PATH]")
	}
	modelPath := fs.String("model", "", "模型 leaves.json 路径（必需）")
	kind := fs.String("type", "importance", "importance（无需数据）| shap（需 --data）")
	dataPath := fs.String("data", "", "数据路径（--type shap 必需）")
	maxRows := fs.Int("max-rows", 100, "SHAP 最多样本数（默认 100，上限 500）")
	metricsPath := fs.String("metrics", "", "输出 JSON（空=stdout）")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *modelPath == "" {
		return errUsage("--model 必需")
	}
	ir, _, err := leavesio.ParseLeavesJSONFile(*modelPath)
	if err != nil {
		return fmt.Errorf("parse model: %w", err)
	}
	if ir.Forest == nil {
		return fmt.Errorf("model has no forest（仅树模型支持 explain）")
	}

	switch *kind {
	case "importance":
		return explainImportance(ir, *dataPath, *metricsPath)
	case "shap":
		return explainSHAP(ir, *dataPath, *maxRows, *metricsPath)
	default:
		return errUsage("--type 仅支持 importance 或 shap")
	}
}

func explainImportance(ir *model.ModelIR, dataPath, metricsPath string) error {
	names := ir.FeatureNames
	if len(names) == 0 && dataPath != "" {
		if dm, err := data.FromFileAuto(dataPath); err == nil {
			if d, ok := dm.(*data.Dense); ok && len(d.FNames) > 0 {
				names = d.FNames
			}
		}
	}
	fi := explain.ComputeImportance(ir.Forest, explain.ImportanceGain, names)
	feats := []map[string]any{}
	nFeat := 0
	if fi != nil {
		nFeat = len(fi.Scores)
		for i := 0; i < nFeat; i++ {
			feats = append(feats, map[string]any{"name": fi.Names[i], "score": fi.Scores[i]})
		}
	}
	return writeJSON(metricsPath, map[string]any{
		"type":       "gain",
		"n_features": nFeat,
		"features":   feats,
	})
}

func explainSHAP(ir *model.ModelIR, dataPath string, maxRows int, metricsPath string) error {
	if dataPath == "" {
		return errUsage("--type shap 需要 --data")
	}
	if ir.NOutputGroups > 1 {
		return fmt.Errorf("multi-class SHAP 暂不支持 CLI explain；用 --type importance 代替")
	}
	dm, err := data.FromFileAuto(dataPath)
	if err != nil {
		return fmt.Errorf("load data: %w", err)
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
	shapRows, err := exp.ShapleyValues(rows)
	if err != nil {
		return fmt.Errorf("shap: %w", err)
	}
	base := exp.ExpectedValue()

	out := make([]map[string]any, len(rows))
	names := ir.FeatureNames
	for i := range rows {
		row := map[string]any{"row": i, "base": base}
		feats := []map[string]any{}
		for j := range shapRows[i] {
			nm := ""
			if j < len(names) {
				nm = names[j]
			}
			feats = append(feats, map[string]any{"name": nm, "value": shapRows[i][j]})
		}
		row["features"] = feats
		out[i] = row
	}
	return writeJSON(metricsPath, out)
}
