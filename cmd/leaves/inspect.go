package main

import (
	"flag"
	"fmt"

	leavesio "github.com/linkerlin/leaves/v2/io"
	"github.com/linkerlin/leaves/v2/model"
)

func cmdInspect(args []string) error {
	fs := flag.NewFlagSet("inspect", flag.ContinueOnError)
	fs.Usage = func() { fmt.Fprintln(fs.Output(), "用法: leaves inspect --model PATH [--metrics PATH]") }
	modelPath := fs.String("model", "", "模型 leaves.json 路径（必需）")
	metricsPath := fs.String("metrics", "", "输出 JSON（空=stdout）")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *modelPath == "" {
		return errUsage("--model 必需")
	}

	ir, objective, err := leavesio.ParseLeavesJSONFile(*modelPath)
	if err != nil {
		return fmt.Errorf("parse model: %w", err)
	}

	kind := "unknown"
	switch ir.Kind {
	case model.KindGBTree:
		kind = "gbtree"
	case model.KindDART:
		kind = "dart"
	case model.KindGBLinear:
		kind = "gblinear"
	case model.KindSklearnGBDT:
		kind = "sklearn_gbdt"
	}
	nTrees := 0
	if ir.Forest != nil {
		nTrees = len(ir.Forest.Trees)
	}

	doc := map[string]any{
		"file":            *modelPath,
		"objective":       objective,
		"kind":            kind,
		"num_features":    ir.NumFeatures,
		"n_output_groups": ir.NOutputGroups,
		"n_trees":         nTrees,
		"name":            ir.Name,
	}
	if len(ir.FeatureNames) > 0 {
		doc["feature_names"] = ir.FeatureNames
	}
	return writeJSON(*metricsPath, doc)
}
