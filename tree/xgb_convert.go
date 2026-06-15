package tree

import (
	"fmt"

	"github.com/linkerlin/leaves/internal/xgbin"
)

// TreeIRFromXGBModel 将 XGBoost 树模型（二进制或 JSON 解析结果）转为 TreeIR。
func TreeIRFromXGBModel(origTree *xgbin.TreeModel, numFeatures uint32) (*TreeIR, error) {
	nodes, leafValues, leafOrigNodes, origIdxs, err := xgbNodesFromTreeModel(origTree, numFeatures)
	if err != nil {
		return nil, err
	}
	t := BuildTreeIR(nodes, leafValues, nil, nil, 0)
	if origTree.Param.SizeLeafVector > 1 {
		t.OutputDim = int(origTree.Param.SizeLeafVector)
	}
	if len(origTree.LeafWeights) > 0 && t.OutputDim > 1 {
		t.LeafValue = xgbVectorLeafValues(origTree, leafOrigNodes, t.OutputDim)
	}
	if origTree.Stats != nil && len(origIdxs) == len(nodes) {
		t.SplitGain = make([]float64, len(nodes))
		t.SumHess = make([]float64, len(nodes))
		for i, oi := range origIdxs {
			if int(oi) < len(origTree.Stats) {
				t.SplitGain[i] = float64(origTree.Stats[oi].LossChg)
				t.SumHess[i] = float64(origTree.Stats[oi].SumHess)
			}
		}
	}
	return t, nil
}

func xgbSplitIndex(origNode *xgbin.Node) uint32 {
	return origNode.SIndex & ((1 << 31) - 1)
}

func xgbDefaultLeft(origNode *xgbin.Node) bool {
	return (origNode.SIndex >> 31) != 0
}

func xgbIsLeaf(origNode *xgbin.Node) bool {
	return origNode.CLeft == -1
}

func xgbVectorLeafValues(origTree *xgbin.TreeModel, leafOrigNodes []int32, dim int) []float64 {
	if dim <= 0 {
		dim = 1
	}
	out := make([]float64, len(leafOrigNodes)*dim)
	for li, nodeIdx := range leafOrigNodes {
		base := int(nodeIdx) * dim
		for d := 0; d < dim; d++ {
			if base+d < len(origTree.LeafWeights) {
				out[li*dim+d] = origTree.LeafWeights[base+d]
			}
		}
	}
	return out
}

func xgbNodesFromTreeModel(origTree *xgbin.TreeModel, numFeatures uint32) ([]LgNodeData, []float64, []int32, []uint32, error) {
	if origTree.Param.NumFeature > int32(numFeatures) {
		return nil, nil, nil, nil, fmt.Errorf(
			"tree number of features %d, but header number of features %d",
			origTree.Param.NumFeature, numFeatures,
		)
	}
	if origTree.Param.NumRoots != 1 {
		return nil, nil, nil, nil, fmt.Errorf("support only trees with 1 root (got %d)", origTree.Param.NumRoots)
	}
	if origTree.Param.NumNodes == 0 {
		return nil, nil, nil, nil, fmt.Errorf("tree with zero number of nodes")
	}

	numNodes := origTree.Param.NumNodes
	leafValues := make([]float64, 0, numNodes)
	leafOrigNodes := make([]int32, 0, numNodes)

	if numNodes == 1 {
		lv := []float64{float64(origTree.Nodes[0].Info)}
		if origTree.Param.SizeLeafVector > 1 && len(origTree.LeafWeights) > 0 {
			lv = xgbVectorLeafValues(origTree, []int32{0}, int(origTree.Param.SizeLeafVector))
		}
		return nil, lv, []int32{0}, nil, nil
	}

	nodes := make([]LgNodeData, 0, numNodes)
	origIdxs := make([]uint32, 0, numNodes)

	createNode := func(origNode *xgbin.Node) (LgNodeData, error) {
		missingType := uint8(flagMissingNan)
		defaultType := uint8(0)
		if xgbDefaultLeft(origNode) {
			defaultType = flagDefaultLeft
		}
		node := LgNodeData{
			Threshold: float64(origNode.Info),
			Feature:   xgbSplitIndex(origNode),
			Flags:     missingType | defaultType,
		}

		if origNode.CLeft < 0 {
			return node, fmt.Errorf("logic error: got origNode.CLeft < 0")
		}
		if origNode.CRight < 0 {
			return node, fmt.Errorf("logic error: got origNode.CRight < 0")
		}
		if origTree.Nodes[origNode.CLeft].CLeft == -1 {
			node.Flags |= flagLeftLeaf
			node.Left = uint32(len(leafValues))
			leafOrigNodes = append(leafOrigNodes, origNode.CLeft)
			leafValues = append(leafValues, float64(origTree.Nodes[origNode.CLeft].Info))
		} else {
			node.Left = uint32(origNode.CLeft)
		}
		if origTree.Nodes[origNode.CRight].CLeft == -1 {
			node.Flags |= flagRightLeaf
			node.Right = uint32(len(leafValues))
			leafOrigNodes = append(leafOrigNodes, origNode.CRight)
			leafValues = append(leafValues, float64(origTree.Nodes[origNode.CRight].Info))
		} else {
			node.Right = uint32(origNode.CRight)
		}
		return node, nil
	}

	origNodeIdxStack := make([]uint32, 0, numNodes)
	convNodeIdxStack := make([]uint32, 0, numNodes)
	visited := make([]bool, numNodes)

	node, err := createNode(&origTree.Nodes[0])
	if err != nil {
		return nil, nil, nil, nil, err
	}
	nodes = append(nodes, node)
	origIdxs = append(origIdxs, 0)
	origNodeIdxStack = append(origNodeIdxStack, 0)
	convNodeIdxStack = append(convNodeIdxStack, 0)

	for len(origNodeIdxStack) > 0 {
		convIdx := convNodeIdxStack[len(convNodeIdxStack)-1]
		if nodes[convIdx].Flags&flagRightLeaf == 0 {
			origIdx := uint32(origTree.Nodes[origNodeIdxStack[len(origNodeIdxStack)-1]].CRight)
			if !visited[origIdx] {
				node, err := createNode(&origTree.Nodes[origIdx])
				if err != nil {
					return nil, nil, nil, nil, err
				}
				nodes = append(nodes, node)
				origIdxs = append(origIdxs, origIdx)
				convNewIdx := len(nodes) - 1
				convNodeIdxStack = append(convNodeIdxStack, uint32(convNewIdx))
				origNodeIdxStack = append(origNodeIdxStack, origIdx)
				visited[origIdx] = true
				nodes[convIdx].Right = uint32(convNewIdx)
				continue
			}
		}
		if nodes[convIdx].Flags&flagLeftLeaf == 0 {
			origIdx := uint32(origTree.Nodes[origNodeIdxStack[len(origNodeIdxStack)-1]].CLeft)
			if !visited[origIdx] {
				node, err := createNode(&origTree.Nodes[origIdx])
				if err != nil {
					return nil, nil, nil, nil, err
				}
				nodes = append(nodes, node)
				origIdxs = append(origIdxs, origIdx)
				convNewIdx := len(nodes) - 1
				convNodeIdxStack = append(convNodeIdxStack, uint32(convNewIdx))
				origNodeIdxStack = append(origNodeIdxStack, origIdx)
				visited[origIdx] = true
				nodes[convIdx].Left = uint32(convNewIdx)
				continue
			}
		}
		origNodeIdxStack = origNodeIdxStack[:len(origNodeIdxStack)-1]
		convNodeIdxStack = convNodeIdxStack[:len(convNodeIdxStack)-1]
	}

	return nodes, leafValues, leafOrigNodes, origIdxs, nil
}
