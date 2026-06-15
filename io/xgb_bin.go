package io

import (
	"bufio"
	"fmt"
	"os"

	"github.com/linkerlin/leaves/internal/xgbin"
	"github.com/linkerlin/leaves/linear"
	"github.com/linkerlin/leaves/model"
	"github.com/linkerlin/leaves/tree"
)

// ParseXGBoostBinaryFile 解析经典 XGBoost 二进制模型（gbtree/dart/gblinear）。
func ParseXGBoostBinaryFile(filename string) (*XGBoostLoadResult, error) {
	f, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return ParseXGBoostBinaryReader(bufio.NewReader(f))
}

// ParseXGBoostBinaryReader 从 reader 解析 XGBoost 二进制模型。
func ParseXGBoostBinaryReader(reader *bufio.Reader) (*XGBoostLoadResult, error) {
	header, err := xgbin.ReadModelHeader(reader)
	if err != nil {
		return nil, err
	}
	if header.Param.NumFeatures == 0 {
		return nil, fmt.Errorf("zero number of features")
	}
	switch header.NameGbm {
	case "gbtree", "dart":
		return parseXGBBinaryGBTree(reader, header)
	case "gblinear":
		return parseXGBBinaryGBLinear(reader, header)
	default:
		return nil, fmt.Errorf("unsupported xgboost binary booster %q", header.NameGbm)
	}
}

func parseXGBBinaryGBTree(reader *bufio.Reader, header *xgbin.ModelHeader) (*XGBoostLoadResult, error) {
	origModel, err := xgbin.ReadGBTreeModel(reader)
	if err != nil {
		return nil, err
	}
	if origModel.Param.NumFeature > int32(header.Param.NumFeatures) {
		return nil, fmt.Errorf("gbtree num_feature %d > header %d",
			origModel.Param.NumFeature, header.Param.NumFeatures)
	}

	numOutputGroups := int(origModel.Param.NumOutputGroup)
	if numOutputGroups <= 0 {
		numOutputGroups = 1
	}
	if header.Param.NumClass > 0 {
		numOutputGroups = int(header.Param.NumClass)
	}

	weightDrop := make([]float64, origModel.Param.NumTrees)
	for i := range weightDrop {
		weightDrop[i] = 1.0
	}
	if header.NameGbm == "dart" {
		wd, err := xgbin.ReadFloat32Slice(reader)
		if err != nil {
			return nil, err
		}
		if len(wd) != int(origModel.Param.NumTrees) {
			return nil, fmt.Errorf("dart weight_drop len %d != num_trees %d", len(wd), origModel.Param.NumTrees)
		}
		for i, v := range wd {
			weightDrop[i] = float64(v)
		}
	}

	numFeatures := int(header.Param.NumFeatures)
	trees := make([]tree.TreeIR, 0, origModel.Param.NumTrees)
	for i := int32(0); i < origModel.Param.NumTrees; i++ {
		tir, err := tree.TreeIRFromXGBModel(origModel.Trees[i], header.Param.NumFeatures)
		if err != nil {
			return nil, fmt.Errorf("tree %d: %w", i, err)
		}
		trees = append(trees, *tir)
	}

	treeInfo := make([]int, len(origModel.TreeInfo))
	for i, v := range origModel.TreeInfo {
		treeInfo[i] = int(v)
	}
	if len(treeInfo) == 0 {
		treeInfo = make([]int, len(trees))
		for i := range treeInfo {
			treeInfo[i] = i % numOutputGroups
		}
	}

	iterationIndptr := make([]int, len(trees)/numOutputGroups+1)
	for i := range iterationIndptr {
		iterationIndptr[i] = i * numOutputGroups
	}

	name := "xgboost.gbtree"
	kind := model.KindGBTree
	if header.NameGbm == "dart" {
		name = "xgboost.dart"
		kind = model.KindDART
	}

	forest := &tree.ForestIR{
		NumFeatures:       numFeatures,
		NumOutputGroups:   numOutputGroups,
		BaseScore:         float64(header.Param.BaseScore),
		TreeInfo:          treeInfo,
		IterationIndptr:   iterationIndptr,
		Trees:             trees,
		WeightDrop:        weightDrop,
		Name:              name,
	}

	return &XGBoostLoadResult{
		IR: &model.ModelIR{
			Kind:             kind,
			NumFeatures:      numFeatures,
			NRawOutputGroups: numOutputGroups,
			NOutputGroups:    numOutputGroups,
			Name:             name,
			Forest:           forest,
		},
		Objective: header.NameObj,
	}, nil
}

func parseXGBBinaryGBLinear(reader *bufio.Reader, header *xgbin.ModelHeader) (*XGBoostLoadResult, error) {
	gbModel, err := xgbin.ReadGBLinearModel(reader)
	if err != nil {
		return nil, err
	}
	nf := int(gbModel.Param.NumFeature)
	ng := int(gbModel.Param.NumOutputGroup)
	if ng <= 0 {
		ng = 1
	}
	weights := make([]float64, len(gbModel.Weights))
	for i, w := range gbModel.Weights {
		weights[i] = float64(w)
	}
	lin := &linear.LinearIR{
		NumFeatures:     nf,
		NumOutputGroups: ng,
		BaseScore:       float64(header.Param.BaseScore),
		Weights:         weights,
		Name:            "xgboost.gblinear",
	}
	adjustLinearBaseScoreForObjective(header.NameObj, lin)

	return &XGBoostLoadResult{
		IR: &model.ModelIR{
			Kind:             model.KindGBLinear,
			NumFeatures:      nf,
			NRawOutputGroups: ng,
			NOutputGroups:    ng,
			Name:             "xgboost.gblinear",
			Linear:           lin,
		},
		Objective: header.NameObj,
	}, nil
}
