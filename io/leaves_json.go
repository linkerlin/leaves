package io

import (
	"encoding/json"
	"fmt"
	"io"
	"os"

	"github.com/linkerlin/leaves/linear"
	"github.com/linkerlin/leaves/model"
	"github.com/linkerlin/leaves/tree"
)

const leavesFormatVersion = 1

// LeavesJSON 训练产出格式（与 LoadFromFile 对称）。
type LeavesJSON struct {
	LeavesVersion int    `json:"leaves_version"`
	Objective     string `json:"objective"`
	Model         leavesModelJSON `json:"model"`
}

type leavesModelJSON struct {
	Kind             int          `json:"kind"`
	NumFeatures      int          `json:"num_features"`
	NRawOutputGroups int          `json:"n_raw_output_groups"`
	NOutputGroups    int          `json:"n_output_groups"`
	Name             string       `json:"name"`
	Forest           *forestJSON  `json:"forest,omitempty"`
	Linear           *linearJSON  `json:"linear,omitempty"`
}

type linearJSON struct {
	NumFeatures     int       `json:"num_features"`
	NumOutputGroups int       `json:"num_output_groups"`
	BaseScore       float64   `json:"base_score"`
	Weights         []float64 `json:"weights"`
	Name            string    `json:"name"`
}

type forestJSON struct {
	NumFeatures       int         `json:"num_features"`
	NumOutputGroups   int         `json:"num_output_groups"`
	BaseScore         float64     `json:"base_score"`
	BaseScores        []float64   `json:"base_scores,omitempty"`
	AverageOutput     bool        `json:"average_output"`
	Name              string      `json:"name"`
	NumParallelTree   int         `json:"num_parallel_tree,omitempty"`
	IterationIndptr   []int       `json:"iteration_indptr,omitempty"`
	TreeInfo          []int       `json:"tree_info,omitempty"`
	WeightDrop        []float64   `json:"weight_drop,omitempty"`
	Trees             []treeJSON  `json:"trees"`
}

type treeJSON struct {
	NumLeaves      int       `json:"num_leaves"`
	NumNodes       int       `json:"num_nodes"`
	MaxDepth       int       `json:"max_depth"`
	OutputDim      int       `json:"output_dim,omitempty"`
	SplitFeature   []int32   `json:"split_feature"`
	SplitThreshold []float64 `json:"split_threshold"`
	DefaultLeft    []bool    `json:"default_left"`
	MissingZero    []bool    `json:"missing_zero"`
	MissingNan     []bool    `json:"missing_nan"`
	LeftChild      []int32   `json:"left_child"`
	RightChild     []int32   `json:"right_child"`
	LeafValue      []float64 `json:"leaf_value"`
	SplitGain      []float64 `json:"split_gain,omitempty"`
	SumHess        []float64 `json:"sum_hess,omitempty"`
}

// LeavesLoadResult leaves.json 加载结果。
type LeavesLoadResult struct {
	IR        *model.ModelIR
	Objective string
}

// LoadLeavesJSONFile 加载 leaves.json 并返回结果。
func LoadLeavesJSONFile(path string) (*LeavesLoadResult, error) {
	ir, obj, err := ParseLeavesJSONFile(path)
	if err != nil {
		return nil, err
	}
	return &LeavesLoadResult{IR: ir, Objective: obj}, nil
}

// SaveLeavesJSON 保存训练模型。
func SaveLeavesJSON(w io.Writer, ir *model.ModelIR, objective string) error {
	if ir == nil {
		return fmt.Errorf("leaves json: nil model")
	}
	doc := LeavesJSON{
		LeavesVersion: leavesFormatVersion,
		Objective:     objective,
		Model: leavesModelJSON{
			Kind:             int(ir.Kind),
			NumFeatures:      ir.NumFeatures,
			NRawOutputGroups: ir.NRawOutputGroups,
			NOutputGroups:    ir.NOutputGroups,
			Name:             ir.Name,
		},
	}
	if ir.Forest != nil {
		doc.Model.Forest = forestToJSON(ir.Forest)
	}
	if ir.Linear != nil {
		doc.Model.Linear = linearToJSON(ir.Linear)
	}
	if doc.Model.Forest == nil && doc.Model.Linear == nil {
		return fmt.Errorf("leaves json: empty model")
	}
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	return enc.Encode(doc)
}

// SaveLeavesJSONFile 保存到文件。
func SaveLeavesJSONFile(path string, ir *model.ModelIR, objective string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	return SaveLeavesJSON(f, ir, objective)
}

// ParseLeavesJSON 解析 leaves.json。
func ParseLeavesJSON(r io.Reader) (*model.ModelIR, string, error) {
	var doc LeavesJSON
	if err := json.NewDecoder(r).Decode(&doc); err != nil {
		return nil, "", fmt.Errorf("leaves json: %w", err)
	}
	if doc.LeavesVersion != leavesFormatVersion {
		return nil, "", fmt.Errorf("leaves json: unsupported version %d", doc.LeavesVersion)
	}
	ir := &model.ModelIR{
		Kind:             model.ModelKind(doc.Model.Kind),
		NumFeatures:      doc.Model.NumFeatures,
		NRawOutputGroups: doc.Model.NRawOutputGroups,
		NOutputGroups:    doc.Model.NOutputGroups,
		Name:             doc.Model.Name,
	}
	if doc.Model.Forest != nil {
		ir.Forest = forestFromJSON(doc.Model.Forest)
	}
	if doc.Model.Linear != nil {
		ir.Linear = linearFromJSON(doc.Model.Linear)
	}
	if ir.Forest == nil && ir.Linear == nil {
		return nil, "", fmt.Errorf("leaves json: missing booster data")
	}
	return ir, doc.Objective, nil
}

// ParseLeavesJSONFile 从文件解析。
func ParseLeavesJSONFile(path string) (*model.ModelIR, string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, "", err
	}
	defer f.Close()
	return ParseLeavesJSON(f)
}

func forestToJSON(f *tree.ForestIR) *forestJSON {
	out := &forestJSON{
		NumFeatures:     f.NumFeatures,
		NumOutputGroups: f.NumOutputGroups,
		BaseScore:       f.BaseScore,
		BaseScores:      f.BaseScores,
		AverageOutput:   f.AverageOutput,
		Name:            f.Name,
		NumParallelTree: f.NumParallelTree,
		IterationIndptr: f.IterationIndptr,
		TreeInfo:        f.TreeInfo,
		WeightDrop:      f.WeightDrop,
		Trees:           make([]treeJSON, len(f.Trees)),
	}
	for i, t := range f.Trees {
		out.Trees[i] = treeToJSON(&t)
	}
	return out
}

func treeToJSON(t *tree.TreeIR) treeJSON {
	return treeJSON{
		NumLeaves:      t.NumLeaves,
		NumNodes:       t.NumNodes,
		MaxDepth:       t.MaxDepth,
		OutputDim:      t.OutputDim,
		SplitFeature:   t.SplitFeature,
		SplitThreshold: t.SplitThreshold,
		DefaultLeft:    t.DefaultLeft,
		MissingZero:    t.MissingZero,
		MissingNan:     t.MissingNan,
		LeftChild:      t.LeftChild,
		RightChild:     t.RightChild,
		LeafValue:      t.LeafValue,
		SplitGain:      t.SplitGain,
		SumHess:        t.SumHess,
	}
}

func forestFromJSON(f *forestJSON) *tree.ForestIR {
	out := &tree.ForestIR{
		NumFeatures:     f.NumFeatures,
		NumOutputGroups: f.NumOutputGroups,
		BaseScore:       f.BaseScore,
		BaseScores:      f.BaseScores,
		AverageOutput:   f.AverageOutput,
		Name:            f.Name,
		NumParallelTree: f.NumParallelTree,
		IterationIndptr: f.IterationIndptr,
		TreeInfo:        f.TreeInfo,
		WeightDrop:      f.WeightDrop,
		Trees:           make([]tree.TreeIR, len(f.Trees)),
	}
	for i, t := range f.Trees {
		out.Trees[i] = treeFromJSON(&t)
	}
	return out
}

func treeFromJSON(t *treeJSON) tree.TreeIR {
	return tree.TreeIR{
		NumLeaves:      t.NumLeaves,
		NumNodes:       t.NumNodes,
		MaxDepth:       t.MaxDepth,
		OutputDim:      t.OutputDim,
		SplitFeature:   t.SplitFeature,
		SplitThreshold: t.SplitThreshold,
		DefaultLeft:    t.DefaultLeft,
		MissingZero:    t.MissingZero,
		MissingNan:     t.MissingNan,
		LeftChild:      t.LeftChild,
		RightChild:     t.RightChild,
		LeafValue:      t.LeafValue,
		SplitGain:      t.SplitGain,
		SumHess:        t.SumHess,
	}
}

func linearToJSON(l *linear.LinearIR) *linearJSON {
	return &linearJSON{
		NumFeatures:     l.NumFeatures,
		NumOutputGroups: l.NumOutputGroups,
		BaseScore:       l.BaseScore,
		Weights:         l.Weights,
		Name:            l.Name,
	}
}

func linearFromJSON(l *linearJSON) *linear.LinearIR {
	return &linear.LinearIR{
		NumFeatures:     l.NumFeatures,
		NumOutputGroups: l.NumOutputGroups,
		BaseScore:       l.BaseScore,
		Weights:         l.Weights,
		Name:            l.Name,
	}
}
