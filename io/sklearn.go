package io

import (
	"bufio"
	"fmt"
	"os"

	"github.com/linkerlin/leaves/v2/internal/pickle"
	"github.com/linkerlin/leaves/v2/model"
	"github.com/linkerlin/leaves/v2/tree"
)

// ParseSklearnPickleFile 解析 sklearn GradientBoosting pickle（实验协议）为 ForestIR。
func ParseSklearnPickleFile(filename string) (*LeavesLoadResult, error) {
	f, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return ParseSklearnPickleReader(bufio.NewReader(f))
}

// ParseSklearnPickleReader 从 reader 解析 sklearn GB pickle。
func ParseSklearnPickleReader(r *bufio.Reader) (*LeavesLoadResult, error) {
	dec := pickle.NewDecoder(r)
	res, err := dec.Decode()
	if err != nil {
		return nil, fmt.Errorf("error while decoding: %s", err.Error())
	}
	gbdt := pickle.SklearnGradientBoosting{}
	if err := pickle.ParseClass(&gbdt, res); err != nil {
		return nil, fmt.Errorf("error while parsing gradient boosting class: %s", err.Error())
	}

	nGroups := gbdt.NClasses
	if nGroups == 2 {
		nGroups = 1
	}
	if gbdt.NEstimators == 0 {
		return nil, fmt.Errorf("no trees in file")
	}
	if gbdt.NEstimators*nGroups != len(gbdt.Estimators) {
		return nil, fmt.Errorf("unexpected number of trees (NEstimators = %d, nRawOutputGroups = %d, len(Estimators) = %d)",
			gbdt.NEstimators, nGroups, len(gbdt.Estimators))
	}

	scale := gbdt.LearningRate
	base := make([]float64, nGroups)
	switch gbdt.InitEstimator.Name {
	case "LogOddsEstimator":
		for i := 0; i < nGroups; i++ {
			base[i] = gbdt.InitEstimator.Prior[0]
		}
	case "PriorProbabilityEstimator":
		if len(gbdt.InitEstimator.Prior) != len(base) {
			return nil, fmt.Errorf("len(gbdt.InitEstimator.Prior) != len(base)")
		}
		base = gbdt.InitEstimator.Prior
	default:
		return nil, fmt.Errorf("unknown initial estimator %q", gbdt.InitEstimator.Name)
	}

	trees := make([]tree.TreeIR, 0, gbdt.NEstimators*nGroups)
	for i := 0; i < gbdt.NEstimators; i++ {
		for j := 0; j < nGroups; j++ {
			treeNum := i*nGroups + j
			tir, err := sklearnTreeToIR(gbdt.Estimators[treeNum], scale, base[j])
			if err != nil {
				return nil, fmt.Errorf("error while creating %d tree: %s", treeNum, err.Error())
			}
			trees = append(trees, *tir)
		}
		for k := range base {
			base[k] = 0
		}
	}

	nFeat := gbdt.MaxFeatures
	if nFeat < 0 {
		nFeat = 0
	}
	name := "sklearn.ensemble.GradientBoostingClassifier"
	forest := lgbForestIR(name, nFeat, nGroups, trees, false)
	return &LeavesLoadResult{
		IR: &model.ModelIR{
			Kind:             model.KindSklearnGBDT,
			NumFeatures:      nFeat,
			NRawOutputGroups: nGroups,
			NOutputGroups:    nGroups,
			Name:             name,
			Forest:           forest,
		},
	}, nil
}

func sklearnTreeToIR(sk pickle.SklearnDecisionTreeRegressor, scale, base float64) (*tree.TreeIR, error) {
	numLeaves := 0
	numNodes := 0
	for _, n := range sk.Tree.Nodes {
		if n.LeftChild < 0 {
			numLeaves++
		} else {
			numNodes++
		}
	}
	if numLeaves-1 != numNodes {
		return nil, fmt.Errorf("unexpected number of leaves (%d) and nodes (%d)", numLeaves, numNodes)
	}

	if numNodes == 0 {
		v := 0.0
		if len(sk.Tree.Values) > 0 {
			v = sk.Tree.Values[0]*scale + base
		}
		return tree.BuildTreeIR(nil, []float64{v}, nil, nil, 0), nil
	}

	var t lgbParsedTree
	createNode := func(idx int) (tree.LgNodeData, error) {
		ref := &sk.Tree.Nodes[idx]
		node := tree.LgNodeData{
			Feature:   uint32(ref.Feature),
			Threshold: ref.Threshold,
		}
		if sk.Tree.Nodes[ref.LeftChild].LeftChild < 0 {
			node.Flags |= lgbFlagLeftLeaf
			node.Left = uint32(len(t.leafValues))
			t.leafValues = append(t.leafValues, sk.Tree.Values[ref.LeftChild]*scale+base)
		}
		if sk.Tree.Nodes[ref.RightChild].LeftChild < 0 {
			node.Flags |= lgbFlagRightLeaf
			node.Right = uint32(len(t.leafValues))
			t.leafValues = append(t.leafValues, sk.Tree.Values[ref.RightChild]*scale+base)
		}
		return node, nil
	}

	origNodeIdxStack := make([]uint32, 0, numNodes)
	convNodeIdxStack := make([]uint32, 0, numNodes)
	visited := make([]bool, sk.Tree.NNodes)
	t.nodes = make([]tree.LgNodeData, 0, numNodes)
	node, err := createNode(0)
	if err != nil {
		return nil, err
	}
	t.nodes = append(t.nodes, node)
	origNodeIdxStack = append(origNodeIdxStack, 0)
	convNodeIdxStack = append(convNodeIdxStack, 0)
	for len(origNodeIdxStack) > 0 {
		convIdx := convNodeIdxStack[len(convNodeIdxStack)-1]
		if t.nodes[convIdx].Flags&lgbFlagRightLeaf == 0 {
			origIdx := sk.Tree.Nodes[origNodeIdxStack[len(origNodeIdxStack)-1]].RightChild
			if !visited[origIdx] {
				node, err := createNode(origIdx)
				if err != nil {
					return nil, err
				}
				t.nodes = append(t.nodes, node)
				convNewIdx := len(t.nodes) - 1
				convNodeIdxStack = append(convNodeIdxStack, uint32(convNewIdx))
				origNodeIdxStack = append(origNodeIdxStack, uint32(origIdx))
				visited[origIdx] = true
				t.nodes[convIdx].Right = uint32(convNewIdx)
				continue
			}
		}
		if t.nodes[convIdx].Flags&lgbFlagLeftLeaf == 0 {
			origIdx := sk.Tree.Nodes[origNodeIdxStack[len(origNodeIdxStack)-1]].LeftChild
			if !visited[origIdx] {
				node, err := createNode(origIdx)
				if err != nil {
					return nil, err
				}
				t.nodes = append(t.nodes, node)
				convNewIdx := len(t.nodes) - 1
				convNodeIdxStack = append(convNodeIdxStack, uint32(convNewIdx))
				origNodeIdxStack = append(origNodeIdxStack, uint32(origIdx))
				visited[origIdx] = true
				t.nodes[convIdx].Left = uint32(convNewIdx)
				continue
			}
		}
		origNodeIdxStack = origNodeIdxStack[:len(origNodeIdxStack)-1]
		convNodeIdxStack = convNodeIdxStack[:len(convNodeIdxStack)-1]
	}
	return t.toIR(), nil
}
