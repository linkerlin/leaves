package main

import (
	"flag"
	"fmt"

	leavesio "github.com/linkerlin/leaves/v2/io"
	"github.com/linkerlin/leaves/v2/model"
	"github.com/linkerlin/leaves/v2/tree"
)

func cmdInspect(args []string) error {
	fs := flag.NewFlagSet("inspect", flag.ContinueOnError)
	fs.Usage = func() { fmt.Fprintln(fs.Output(), "用法: leaves inspect --model PATH [--metrics PATH]") }
	modelPath := fs.String("model", "", "模型路径（必需）")
	metricsPath := fs.String("metrics", "", "输出 JSON（空=stdout）")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *modelPath == "" {
		return errUsage("--model 必需")
	}

	ir, objective, format, ov, err := loadModelIRWithQuant(*modelPath)
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

	formatName := leavesio.SupportOf(format).Name
	if ov != nil {
		formatName = "leaves.quant.json"
	}

	doc := map[string]any{
		"schema_version":      metricsSchemaVersion,
		"file":                *modelPath,
		"format":              formatName,
		"objective":           objective,
		"kind":                kind,
		"num_features":        ir.NumFeatures,
		"n_output_groups":     ir.NOutputGroups,
		"n_raw_output_groups": ir.NRawOutputGroups,
		"n_trees":             nTrees,
		"name":                ir.Name,
	}
	if ov != nil {
		doc["quantized"] = true
		doc["base_model"] = ov.Base
		doc["quant_levels"] = ov.Levels
	}
	if len(ir.FeatureNames) > 0 {
		doc["feature_names"] = ir.FeatureNames
	}
	if len(ir.CategoryMaps) > 0 {
		doc["category_maps"] = ir.CategoryMaps
	}

	if ir.Forest != nil {
		f := ir.Forest
		doc["n_boost_rounds"] = f.NEstimators()
		doc["base_score"] = f.BaseScore
		if len(f.BaseScores) > 0 {
			doc["base_scores"] = f.BaseScores
		}
		if f.NumParallelTree > 1 {
			doc["num_parallel_tree"] = f.NumParallelTree
		}
		maxDepth := 0
		totalNodes := 0
		totalLeaves := 0
		minLeaves := 0
		maxLeaves := 0
		usedFeat := make(map[int]bool)

		for ti := range f.Trees {
			t := &f.Trees[ti]
			d := t.MaxDepth
			if d <= 0 && t.NumNodes > 0 {
				d = computeTreeObservedDepth(t)
			}
			if d > maxDepth {
				maxDepth = d
			}
			totalNodes += t.NumNodes
			totalLeaves += t.NumLeaves
			if ti == 0 || t.NumLeaves < minLeaves {
				minLeaves = t.NumLeaves
			}
			if t.NumLeaves > maxLeaves {
				maxLeaves = t.NumLeaves
			}
			for ni := 0; ni < t.NumNodes && ni < len(t.SplitFeature); ni++ {
				fi := int(t.SplitFeature[ni])
				if fi >= 0 {
					usedFeat[fi] = true
				}
			}
		}

		meanLeaves := 0.0
		if nTrees > 0 {
			meanLeaves = round4(float64(totalLeaves) / float64(nTrees))
		}
		doc["max_depth_observed"] = maxDepth
		doc["total_nodes"] = totalNodes
		doc["total_leaves"] = totalLeaves
		doc["min_leaves"] = minLeaves
		doc["max_leaves"] = maxLeaves
		doc["mean_leaves_per_tree"] = meanLeaves

		if ir.NumFeatures > 0 {
			var unusedNames []string
			for fi := 0; fi < ir.NumFeatures; fi++ {
				if !usedFeat[fi] {
					name := fmt.Sprintf("f%d", fi)
					if fi < len(ir.FeatureNames) && ir.FeatureNames[fi] != "" {
						name = ir.FeatureNames[fi]
					}
					unusedNames = append(unusedNames, name)
				}
			}
			doc["used_features_count"] = len(usedFeat)
			doc["unused_features_count"] = len(unusedNames)
			if len(unusedNames) > 0 {
				doc["unused_features"] = unusedNames
			}
		}
	}

	if ir.Linear != nil {
		doc["base_score"] = ir.Linear.BaseScore
		doc["linear_weights"] = ir.Linear.Weights
	}

	return writeJSON(*metricsPath, doc)
}

func computeTreeObservedDepth(t *tree.TreeIR) int {
	if t == nil || t.NumNodes <= 0 {
		return 0
	}
	visited := make([]bool, t.NumNodes)
	var walk func(node int) int
	walk = func(node int) int {
		if node < 0 || node >= t.NumNodes || visited[node] {
			return 0
		}
		visited[node] = true
		ld, rd := 1, 1
		if node < len(t.LeftChild) && t.LeftChild[node] >= 0 {
			ld = 1 + walk(int(t.LeftChild[node]))
		}
		if node < len(t.RightChild) && t.RightChild[node] >= 0 {
			rd = 1 + walk(int(t.RightChild[node]))
		}
		if ld > rd {
			return ld
		}
		return rd
	}
	return walk(0)
}
