package io

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"strconv"

	"github.com/dmitryikh/leaves/model"
	"github.com/dmitryikh/leaves/tree"
)

// ExportXGBoostJSON 将 ModelIR 导出为 XGBoost 3.x JSON 格式。
func ExportXGBoostJSON(w io.Writer, ir *model.ModelIR, objective string) error {
	if ir == nil || ir.Forest == nil {
		return fmt.Errorf("xgb export: nil forest")
	}
	doc := buildXGBExportDoc(ir, objective)
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	return enc.Encode(doc)
}

// ExportXGBoostJSONFile 导出到文件。
func ExportXGBoostJSONFile(path string, ir *model.ModelIR, objective string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	return ExportXGBoostJSON(f, ir, objective)
}

func buildXGBExportDoc(ir *model.ModelIR, objective string) map[string]interface{} {
	f := ir.Forest
	boosterName := "gbtree"
	if ir.Kind == model.KindDART || f.Name == "leaves.dart" || f.Name == "xgboost.dart" {
		boosterName = "dart"
	}
	numClass := ir.NRawOutputGroups
	if numClass <= 1 {
		numClass = 0
	}
	baseScoreStr := formatBaseScore(f)
	trees := make([]map[string]interface{}, len(f.Trees))
	for i := range f.Trees {
		trees[i] = exportTreeJSON(&f.Trees[i], i)
	}
	modelObj := map[string]interface{}{
		"gbtree_model_param": map[string]string{
			"num_trees":         strconv.Itoa(len(f.Trees)),
			"num_parallel_tree": strconv.Itoa(maxInt(1, f.NumParallelTree)),
		},
		"iteration_indptr": f.IterationIndptr,
		"tree_info":        f.TreeInfo,
		"trees":            trees,
	}
	if boosterName == "dart" && len(f.WeightDrop) > 0 {
		modelObj["weight_drop"] = f.WeightDrop
	}
	names := ir.FeatureNames
	if names == nil {
		names = []string{}
	}
	types := ir.FeatureTypes
	if types == nil {
		types = make([]string, len(names))
		for i := range types {
			types[i] = "float"
		}
	}
	return map[string]interface{}{
		"version": []int{3, 2, 0},
		"learner": map[string]interface{}{
			"attributes":    map[string]interface{}{},
			"feature_names": names,
			"feature_types": types,
			"gradient_booster": map[string]interface{}{
				"name":  boosterName,
				"model": modelObj,
			},
			"learner_model_param": map[string]string{
				"base_score":           baseScoreStr,
				"boost_from_average":   "0",
				"num_class":            strconv.Itoa(numClass),
				"num_feature":          strconv.Itoa(f.NumFeatures),
				"num_target":           strconv.Itoa(maxInt(1, ir.NRawOutputGroups)),
			},
			"objective": map[string]interface{}{
				"name": objective,
			},
		},
	}
}

func formatBaseScore(f *tree.ForestIR) string {
	if len(f.BaseScores) > 0 {
		parts := make([]string, len(f.BaseScores))
		for i, v := range f.BaseScores {
			parts[i] = formatSci(v)
		}
		s := "["
		for i, p := range parts {
			if i > 0 {
				s += ","
			}
			s += p
		}
		return s + "]"
	}
	return formatSci(f.BaseScore)
}

func formatSci(v float64) string {
	return strconv.FormatFloat(v, 'E', -1, 64)
}

func exportTreeJSON(t *tree.TreeIR, id int) map[string]interface{} {
	x := treeIRToXGBFlat(t)
	return map[string]interface{}{
		"id": id,
		"tree_param": map[string]string{
			"num_nodes":        strconv.Itoa(x.numNodes),
			"num_feature":      strconv.Itoa(x.numFeature),
			"num_deleted":      "0",
			"size_leaf_vector": strconv.Itoa(maxInt(1, t.OutputDim)),
		},
		"left_children":       x.left,
		"right_children":      x.right,
		"split_indices":       x.splitIdx,
		"split_conditions":    x.splitCond,
		"split_type":          x.splitType,
		"default_left":        x.defaultLeft,
		"base_weights":        x.baseWeights,
		"categories":          []int32{},
		"categories_nodes":    []int32{},
		"categories_segments": []int64{},
		"categories_sizes":    []int64{},
	}
}

type xgbFlatTree struct {
	numNodes    int
	numFeature  int
	left        []int32
	right       []int32
	splitIdx    []int32
	splitCond   []float64
	splitType   []int
	defaultLeft []int
	baseWeights []float64
}

func treeIRToXGBFlat(t *tree.TreeIR) xgbFlatTree {
	out := xgbFlatTree{}
	if t == nil {
		return out
	}
	if t.NumNodes == 0 {
		lv := 0.0
		if len(t.LeafValue) > 0 {
			lv = t.LeafValue[0]
		}
		out.numNodes = 1
		out.left = []int32{-1}
		out.right = []int32{-1}
		out.splitIdx = []int32{0}
		out.splitCond = []float64{lv}
		out.splitType = []int{0}
		out.defaultLeft = []int{0}
		out.baseWeights = []float64{lv}
		return out
	}

	var left, right []int32
	var splitIdx []int32
	var splitCond []float64
	var splitType []int
	var defaultLeft []int
	var baseWeights []float64
	maxFeat := 0

	var addLeaf func(val float64) int32
	addLeaf = func(val float64) int32 {
		idx := int32(len(left))
		left = append(left, -1)
		right = append(right, -1)
		splitIdx = append(splitIdx, 0)
		splitCond = append(splitCond, val)
		splitType = append(splitType, 0)
		defaultLeft = append(defaultLeft, 0)
		baseWeights = append(baseWeights, val)
		return idx
	}

	var build func(nodeIdx int) int32
	build = func(nodeIdx int) int32 {
		xgbIdx := int32(len(left))
		feat := int(t.SplitFeature[nodeIdx])
		if feat+1 > maxFeat {
			maxFeat = feat + 1
		}
		cond := t.SplitThreshold[nodeIdx]
		dfl := 0
		if nodeIdx < len(t.DefaultLeft) && t.DefaultLeft[nodeIdx] {
			dfl = 1
		}
		st := 0
		if nodeIdx < len(t.IsCategorical) && t.IsCategorical[nodeIdx] {
			st = 1
			if nodeIdx < len(t.CatSmall) && t.CatSmall[nodeIdx] {
				cond = float64(uint32(t.SplitThreshold[nodeIdx]))
			}
		}
		left = append(left, 0)
		right = append(right, 0)
		splitIdx = append(splitIdx, int32(feat))
		splitCond = append(splitCond, cond)
		splitType = append(splitType, st)
		defaultLeft = append(defaultLeft, dfl)

		lc := t.LeftChild[nodeIdx]
		if lc < 0 {
			left[xgbIdx] = addLeaf(leafScalar(t, lc))
		} else {
			left[xgbIdx] = build(int(lc))
		}
		rc := t.RightChild[nodeIdx]
		if rc < 0 {
			right[xgbIdx] = addLeaf(leafScalar(t, rc))
		} else {
			right[xgbIdx] = build(int(rc))
		}
		return xgbIdx
	}

	build(0)
	out.numNodes = len(left)
	out.numFeature = maxFeat
	out.left = left
	out.right = right
	out.splitIdx = splitIdx
	out.splitCond = splitCond
	out.splitType = splitType
	out.defaultLeft = defaultLeft
	out.baseWeights = baseWeights
	return out
}

func leafScalar(t *tree.TreeIR, leafRef int32) float64 {
	if leafRef >= 0 {
		return 0
	}
	idx := int(^leafRef)
	if idx < len(t.LeafValue) {
		return t.LeafValue[idx]
	}
	return 0
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}
