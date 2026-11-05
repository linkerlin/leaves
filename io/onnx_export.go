package io

import (
	"fmt"
	"os"

	"github.com/linkerlin/leaves/v2/model"
)

// ExportONNX 将树模型（ModelIR）导出为 ai.onnx.ml 标准 ONNX ModelProto 字节。
// 支持回归（TreeEnsembleRegressor）、二分类（TreeEnsembleClassifier + LOGISTIC）
// 与多分类（TreeEnsembleClassifier + SOFTMAX）。
func ExportONNX(ir *model.ModelIR, objective string) ([]byte, error) {
	if ir == nil || ir.Forest == nil || len(ir.Forest.Trees) == 0 {
		return nil, fmt.Errorf("onnx export: model has no tree forest")
	}
	f := ir.Forest

	var (
		nodesTreeids      []int64
		nodesNodeids      []int64
		nodesFeatureids   []int64
		nodesValues       []float64
		nodesModes        []string
		nodesTruenodeids  []int64
		nodesFalsenodeids []int64
		leafTreeids       []int64
		leafNodeids       []int64
		leafTargetIds     []int64
		leafWeights       []float64
	)

	nGroups := ir.NOutputGroups
	if nGroups < 1 {
		nGroups = f.NumOutputGroups
	}
	if nGroups < 1 {
		nGroups = 1
	}

	isBinary := objective == "binary:logistic" && nGroups == 1
	isMulti := (objective == "multi:softmax" || objective == "multi:softprob") && nGroups >= 2

	for ti := range f.Trees {
		t := &f.Trees[ti]
		for ni := 0; ni < t.NumNodes; ni++ {
			if ni < len(t.IsCategorical) && t.IsCategorical[ni] {
				return nil, fmt.Errorf("onnx export: tree %d node %d uses categorical split (unsupported in ai.onnx.ml TreeEnsemble subset)", ti, ni)
			}
		}

		wScale := 1.0
		if ti < len(f.WeightDrop) && f.WeightDrop[ti] != 0 {
			wScale = f.WeightDrop[ti]
		}

		targetID := 0
		if ti < len(f.TreeInfo) {
			targetID = f.TreeInfo[ti]
		} else if nGroups > 1 {
			targetID = ti % nGroups
		}
		if isBinary {
			// ONNX TreeEnsembleClassifier 二类 LOGISTIC 约定 class 1 树承载正类 logit
			targetID = 1
		}

		tid64 := int64(ti)
		if t.NumNodes == 0 {
			nodesTreeids = append(nodesTreeids, tid64)
			nodesNodeids = append(nodesNodeids, 0)
			nodesFeatureids = append(nodesFeatureids, 0)
			nodesValues = append(nodesValues, 0)
			nodesModes = append(nodesModes, "LEAF")
			nodesTruenodeids = append(nodesTruenodeids, 0)
			nodesFalsenodeids = append(nodesFalsenodeids, 0)

			lv := 0.0
			if len(t.LeafValue) > 0 {
				lv = t.LeafValue[0] * wScale
			}
			leafTreeids = append(leafTreeids, tid64)
			leafNodeids = append(leafNodeids, 0)
			leafTargetIds = append(leafTargetIds, int64(targetID))
			leafWeights = append(leafWeights, lv)
			continue
		}

		resolveChild := func(c int32) int64 {
			if c >= 0 {
				return int64(c)
			}
			leafIdx := int(^c)
			return int64(t.NumNodes + leafIdx)
		}

		for ni := 0; ni < t.NumNodes; ni++ {
			nodesTreeids = append(nodesTreeids, tid64)
			nodesNodeids = append(nodesNodeids, int64(ni))
			nodesFeatureids = append(nodesFeatureids, int64(t.SplitFeature[ni]))
			nodesValues = append(nodesValues, t.SplitThreshold[ni])
			nodesModes = append(nodesModes, "BRANCH_LEQ")
			nodesTruenodeids = append(nodesTruenodeids, resolveChild(t.LeftChild[ni]))
			nodesFalsenodeids = append(nodesFalsenodeids, resolveChild(t.RightChild[ni]))
		}
		for li, lv := range t.LeafValue {
			leafID := int64(t.NumNodes + li)
			nodesTreeids = append(nodesTreeids, tid64)
			nodesNodeids = append(nodesNodeids, leafID)
			nodesFeatureids = append(nodesFeatureids, 0)
			nodesValues = append(nodesValues, 0)
			nodesModes = append(nodesModes, "LEAF")
			nodesTruenodeids = append(nodesTruenodeids, 0)
			nodesFalsenodeids = append(nodesFalsenodeids, 0)

			leafTreeids = append(leafTreeids, tid64)
			leafNodeids = append(leafNodeids, leafID)
			leafTargetIds = append(leafTargetIds, int64(targetID))
			leafWeights = append(leafWeights, lv*wScale)
		}
	}

	if isBinary {
		baseVals := []float64{0, f.BaseScore}
		return writeONNXClassifierWithBase(
			nodesTreeids, nodesNodeids, nodesFeatureids, nodesTruenodeids, nodesFalsenodeids,
			nodesValues, nodesModes,
			leafTreeids, leafNodeids, leafTargetIds, leafWeights,
			[]int64{0, 1}, baseVals, "SUM", "LOGISTIC",
		), nil
	}

	if isMulti {
		labels := make([]int64, nGroups)
		for i := 0; i < nGroups; i++ {
			labels[i] = int64(i)
		}
		baseVals := make([]float64, nGroups)
		if len(f.BaseScores) == nGroups {
			copy(baseVals, f.BaseScores)
		} else {
			for i := 0; i < nGroups; i++ {
				baseVals[i] = f.BaseScore
			}
		}
		return writeONNXClassifierWithBase(
			nodesTreeids, nodesNodeids, nodesFeatureids, nodesTruenodeids, nodesFalsenodeids,
			nodesValues, nodesModes,
			leafTreeids, leafNodeids, leafTargetIds, leafWeights,
			labels, baseVals, "SUM", "SOFTMAX",
		), nil
	}

	baseVals := make([]float64, nGroups)
	if len(f.BaseScores) == nGroups {
		copy(baseVals, f.BaseScores)
	} else {
		for i := 0; i < nGroups; i++ {
			baseVals[i] = f.BaseScore
		}
	}

	return writeONNXTreeEnsembleRegressor(treeEnsembleAttrs{
		nodesTreeids:      nodesTreeids,
		nodesNodeids:      nodesNodeids,
		nodesFeatureids:   nodesFeatureids,
		nodesValues:       nodesValues,
		nodesModes:        nodesModes,
		nodesTruenodeids:  nodesTruenodeids,
		nodesFalsenodeids: nodesFalsenodeids,
		targetTreeids:     leafTreeids,
		targetNodeids:     leafNodeids,
		targetIds:         leafTargetIds,
		targetWeights:     leafWeights,
		nTargets:          nGroups,
		baseValues:        baseVals,
	}), nil
}

// ExportONNXFile 将树模型导出并写入 ONNX 文件。
func ExportONNXFile(path string, ir *model.ModelIR, objective string) error {
	b, err := ExportONNX(ir, objective)
	if err != nil {
		return err
	}
	return os.WriteFile(path, b, 0o644)
}

func writeONNXClassifierWithBase(
	nodesTreeids, nodesNodeids, nodesFeatureids, nodesTruenodeids, nodesFalsenodeids []int64,
	nodesValues []float64, nodesModes []string,
	classTreeids, classNodeids, classIds []int64, classWeights []float64,
	classLabels []int64, baseValues []float64, aggregate, postTransform string,
) []byte {
	nodeBody := attrInts("nodes_treeids", nodesTreeids)
	nodeBody = append(nodeBody, attrInts("nodes_nodeids", nodesNodeids)...)
	nodeBody = append(nodeBody, attrInts("nodes_featureids", nodesFeatureids)...)
	nodeBody = append(nodeBody, attrFloats("nodes_values", nodesValues)...)
	nodeBody = append(nodeBody, attrStrings("nodes_modes", nodesModes)...)
	nodeBody = append(nodeBody, attrInts("nodes_truenodeids", nodesTruenodeids)...)
	nodeBody = append(nodeBody, attrInts("nodes_falsenodeids", nodesFalsenodeids)...)
	nodeBody = append(nodeBody, attrInts("class_treeids", classTreeids)...)
	nodeBody = append(nodeBody, attrInts("class_nodeids", classNodeids)...)
	nodeBody = append(nodeBody, attrInts("class_ids", classIds)...)
	nodeBody = append(nodeBody, attrFloats("class_weights", classWeights)...)
	nodeBody = append(nodeBody, attrInts("classlabels_int64s", classLabels)...)
	if len(baseValues) > 0 {
		nodeBody = append(nodeBody, attrFloats("base_values", baseValues)...)
	}
	if aggregate != "" {
		nodeBody = append(nodeBody, attrStrings("aggregate_function", []string{aggregate})...)
	}
	if postTransform != "" {
		nodeBody = append(nodeBody, attrStrings("post_transform", []string{postTransform})...)
	}

	nodeBody = pbAppendString(nodeBody, 4, "TreeEnsembleClassifier")
	nodeBody = pbAppendString(nodeBody, 5, "ai.onnx.ml")
	nodeBody = pbAppendString(nodeBody, 1, "X")
	nodeBody = pbAppendString(nodeBody, 2, "Y")

	var graph []byte
	graph = pbAppendBytes(graph, 1, nodeBody)
	var m []byte
	m = pbAppendInt64(m, 1, 8)
	m = pbAppendBytes(m, 7, graph)
	return m
}
