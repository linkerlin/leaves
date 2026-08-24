package io

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/linkerlin/leaves/v2/model"
	"github.com/linkerlin/leaves/v2/tree"
	"github.com/linkerlin/leaves/v2/util"
)

// LightGBMLoadResult LGB text/JSON → ForestIR（与 LeavesLoadResult 同形）。
type LightGBMLoadResult = LeavesLoadResult

const (
	lgbFlagCategorical = 1 << 0
	lgbFlagDefaultLeft = 1 << 1
	lgbFlagLeftLeaf    = 1 << 2
	lgbFlagRightLeaf   = 1 << 3
	lgbFlagMissingZero = 1 << 4
	lgbFlagMissingNan  = 1 << 5
	lgbFlagCatOneHot   = 1 << 6
	lgbFlagCatSmall    = 1 << 7
)

type lgbParsedTree struct {
	nodes         []tree.LgNodeData
	leafValues    []float64
	catBoundaries []uint32
	catThresholds []uint32
	nCategorical  uint32
}

func (t lgbParsedTree) toIR() *tree.TreeIR {
	return tree.BuildTreeIR(t.nodes, t.leafValues, t.catBoundaries, t.catThresholds, t.nCategorical)
}

func lgbForestIR(name string, nFeat, nGroups int, trees []tree.TreeIR, average bool) *tree.ForestIR {
	wd := make([]float64, len(trees))
	for i := range wd {
		wd[i] = 1
	}
	return &tree.ForestIR{
		NumFeatures:     nFeat,
		NumOutputGroups: nGroups,
		Trees:           trees,
		WeightDrop:      wd,
		AverageOutput:   average,
		Name:            name,
	}
}

func lgbResult(name string, nFeat, nGroups int, trees []tree.TreeIR, average bool, objective string) *LeavesLoadResult {
	forest := lgbForestIR(name, nFeat, nGroups, trees, average)
	return &LeavesLoadResult{
		IR: &model.ModelIR{
			Kind:             model.KindGBTree,
			NumFeatures:      nFeat,
			NRawOutputGroups: nGroups,
			NOutputGroups:    nGroups,
			Name:             name,
			Forest:           forest,
		},
		Objective: objective,
	}
}

// ParseLightGBMTextFile 解析 LightGBM text 模型为 ForestIR（不依赖根包 init）。
func ParseLightGBMTextFile(filename string) (*LightGBMLoadResult, error) {
	f, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return ParseLightGBMTextReader(bufio.NewReader(f))
}

// ParseLightGBMTextReader 从 reader 解析 LightGBM text。
func ParseLightGBMTextReader(reader *bufio.Reader) (*LightGBMLoadResult, error) {
	params, err := util.ReadParamsUntilBlank(reader)
	if err != nil {
		return nil, err
	}
	if err := params.Compare("version", "v2"); err != nil {
		if err := params.Compare("version", "v3"); err != nil {
			return nil, err
		}
	}
	nClasses, err := params.ToInt("num_class")
	if err != nil {
		return nil, err
	}
	nTreePerIteration, err := params.ToInt("num_tree_per_iteration")
	if err != nil {
		return nil, err
	}
	if nClasses != nTreePerIteration {
		return nil, fmt.Errorf("meet case when num_class (%d) != num_tree_per_iteration (%d)", nClasses, nTreePerIteration)
	} else if nClasses < 1 {
		return nil, fmt.Errorf("num_class (%d) should be > 0", nClasses)
	}
	maxFeatureIdx, err := params.ToInt("max_feature_idx")
	if err != nil {
		return nil, err
	}
	name := "lightgbm.gbdt"
	average := false
	if params.Contains("average_output") {
		name = "lightgbm.rf"
		average = true
	}
	treeSizesStr, err := params.ToString("tree_sizes")
	if err != nil {
		return nil, fmt.Errorf("no tree_sizes field")
	}
	treeSizes := strings.Split(treeSizesStr, " ")
	nTrees := len(treeSizes)
	if nTrees == 0 {
		return nil, fmt.Errorf("no trees in file (based on tree_sizes value)")
	} else if nTrees%nClasses != 0 {
		return nil, fmt.Errorf("wrong number of trees (%d) for number of class (%d)", nTrees, nClasses)
	}

	objective := ""
	if !average {
		if s, err := params.ToString("objective"); err == nil {
			objective = lgbObjectiveName(s)
		}
	}
	trees := make([]tree.TreeIR, 0, nTrees)
	for i := 0; i < nTrees; i++ {
		pt, err := lgbTreeFromReader(reader)
		if err != nil {
			return nil, fmt.Errorf("error while reading %d tree: %s", i, err.Error())
		}
		trees = append(trees, *pt.toIR())
	}
	nFeat := 0
	if maxFeatureIdx > 0 {
		nFeat = maxFeatureIdx + 1
	}
	return lgbResult(name, nFeat, nClasses, trees, average, objective), nil
}

func lgbObjectiveName(objectiveStr string) string {
	if objectiveStr == "poisson" || objectiveStr == "gamma" || objectiveStr == "tweedie" {
		if objectiveStr == "gamma" {
			return "reg:gamma"
		}
		if objectiveStr == "tweedie" {
			return "reg:tweedie"
		}
		return "count:poisson"
	}
	if strings.HasPrefix(objectiveStr, "regression") {
		return "reg:squarederror"
	}
	obj, err := lgbParseObjective(objectiveStr)
	if err != nil {
		return ""
	}
	if obj.name == "binary" && obj.param == "sigmoid" {
		return "binary:logistic"
	}
	if obj.name == "multiclass" && obj.param == "num_class" {
		return "multi:softprob"
	}
	return ""
}

type lgbObj struct {
	name  string
	param string
	value int
}

func lgbParseObjective(objective string) (lgbObj, error) {
	tokens := strings.Split(objective, " ")
	var out lgbObj
	errMsg := fmt.Errorf("unexpected objective field: '%s'", objective)
	if len(tokens) != 2 {
		return out, errMsg
	}
	out.name = tokens[0]
	paramTokens := strings.Split(tokens[1], ":")
	if len(paramTokens) != 2 {
		return out, errMsg
	}
	out.param = paramTokens[0]
	v, err := strconv.Atoi(paramTokens[1])
	if err != nil {
		return out, errMsg
	}
	out.value = v
	return out, nil
}

func lgbConvertMissingType(decisionType uint32) (uint8, error) {
	missingTypeOrig := (decisionType >> 2) & 3
	switch missingTypeOrig {
	case 0:
		return 0, nil
	case 1:
		return lgbFlagMissingZero, nil
	case 2:
		return lgbFlagMissingNan, nil
	default:
		return 0, fmt.Errorf("unknown missing type = %d", missingTypeOrig)
	}
}

func lgbTreeFromReader(reader *bufio.Reader) (lgbParsedTree, error) {
	t := lgbParsedTree{}
	params, err := util.ReadParamsUntilBlank(reader)
	if err != nil {
		return t, err
	}
	numCategorical, err := params.ToInt("num_cat")
	if err != nil {
		return t, err
	}
	t.nCategorical = uint32(numCategorical)
	numLeaves, err := params.ToInt("num_leaves")
	if err != nil {
		return t, err
	}
	if numLeaves < 1 {
		return t, fmt.Errorf("num_leaves < 1")
	}
	numNodes := numLeaves - 1
	leafValues, err := params.ToFloat64Slice("leaf_value")
	if err != nil {
		return t, err
	}
	t.leafValues = leafValues
	if numLeaves == 1 {
		return t, nil
	}
	leftChilds, err := params.ToInt32Slice("left_child")
	if err != nil {
		return t, err
	}
	rightChilds, err := params.ToInt32Slice("right_child")
	if err != nil {
		return t, err
	}
	decisionTypes, err := params.ToUint32Slice("decision_type")
	if err != nil {
		return t, err
	}
	splitFeatures, err := params.ToUint32Slice("split_feature")
	if err != nil {
		return t, err
	}
	thresholds, err := params.ToFloat64Slice("threshold")
	if err != nil {
		return t, err
	}
	catThresholds := make([]uint32, 0)
	catBoundaries := make([]uint32, 0)
	if numCategorical > 0 {
		t.catBoundaries = make([]uint32, 1)
		catThresholds, err = params.ToUint32Slice("cat_threshold")
		if err != nil {
			return t, err
		}
		catBoundaries, err = params.ToUint32Slice("cat_boundaries")
		if err != nil {
			return t, err
		}
	}

	createNumericalNode := func(idx int32) (tree.LgNodeData, error) {
		var node tree.LgNodeData
		missingType, err := lgbConvertMissingType(decisionTypes[idx])
		if err != nil {
			return node, err
		}
		defaultType := uint8(0)
		if decisionTypes[idx]&(1<<1) > 0 {
			defaultType = lgbFlagDefaultLeft
		}
		node.Feature = splitFeatures[idx]
		node.Flags = missingType | defaultType
		node.Threshold = thresholds[idx]
		if leftChilds[idx] < 0 {
			node.Flags |= lgbFlagLeftLeaf
			node.Left = uint32(^leftChilds[idx])
		}
		if rightChilds[idx] < 0 {
			node.Flags |= lgbFlagRightLeaf
			node.Right = uint32(^rightChilds[idx])
		}
		return node, nil
	}
	createCategoricalNode := func(idx int32) (tree.LgNodeData, error) {
		var node tree.LgNodeData
		missingType, err := lgbConvertMissingType(decisionTypes[idx])
		if err != nil {
			return node, err
		}
		catIdx := uint32(thresholds[idx])
		catType := uint8(0)
		bitsetSize := catBoundaries[catIdx+1] - catBoundaries[catIdx]
		thresholdSlice := catThresholds[catBoundaries[catIdx]:catBoundaries[catIdx+1]]
		nBits := util.NumberOfSetBits(thresholdSlice)
		if nBits == 0 {
			return node, fmt.Errorf("no bits set")
		} else if nBits == 1 {
			i, err := util.FirstNonZeroBit(thresholdSlice)
			if err != nil {
				return node, fmt.Errorf("not reached error")
			}
			catIdx = i
			catType = lgbFlagCatOneHot
		} else if bitsetSize == 1 {
			catIdx = catThresholds[catBoundaries[catIdx]]
			catType = lgbFlagCatSmall
		} else {
			catIdx = uint32(len(t.catBoundaries) - 1)
			t.catThresholds = append(t.catThresholds, thresholdSlice...)
			t.catBoundaries = append(t.catBoundaries, uint32(len(t.catThresholds)))
		}
		node.Feature = splitFeatures[idx]
		node.Flags = lgbFlagCategorical | missingType | catType
		node.Threshold = float64(catIdx)
		if leftChilds[idx] < 0 {
			node.Flags |= lgbFlagLeftLeaf
			node.Left = uint32(^leftChilds[idx])
		}
		if rightChilds[idx] < 0 {
			node.Flags |= lgbFlagRightLeaf
			node.Right = uint32(^rightChilds[idx])
		}
		return node, nil
	}
	createNode := func(idx int32) (tree.LgNodeData, error) {
		if decisionTypes[idx]&1 > 0 {
			return createCategoricalNode(idx)
		}
		return createNumericalNode(idx)
	}

	origNodeIdxStack := make([]uint32, 0, numNodes)
	convNodeIdxStack := make([]uint32, 0, numNodes)
	visited := make([]bool, numNodes)
	t.nodes = make([]tree.LgNodeData, 0, numNodes)
	node, err := createNode(0)
	if err != nil {
		return t, err
	}
	t.nodes = append(t.nodes, node)
	origNodeIdxStack = append(origNodeIdxStack, 0)
	convNodeIdxStack = append(convNodeIdxStack, 0)
	for len(origNodeIdxStack) > 0 {
		convIdx := convNodeIdxStack[len(convNodeIdxStack)-1]
		if t.nodes[convIdx].Flags&lgbFlagRightLeaf == 0 {
			origIdx := rightChilds[origNodeIdxStack[len(origNodeIdxStack)-1]]
			if !visited[origIdx] {
				node, err := createNode(origIdx)
				if err != nil {
					return t, err
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
			origIdx := leftChilds[origNodeIdxStack[len(origNodeIdxStack)-1]]
			if !visited[origIdx] {
				node, err := createNode(origIdx)
				if err != nil {
					return t, err
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
	return t, nil
}
