package io

import (
	"encoding/json"
	"fmt"
	io "io"
	"os"
	"strconv"
	"strings"

	"github.com/linkerlin/leaves/v2/tree"
	"github.com/linkerlin/leaves/v2/util"
)

type lgbEnsembleJSON struct {
	Name                 string            `json:"name"`
	Version              string            `json:"version"`
	NumClasses           int               `json:"num_class"`
	NumTreesPerIteration int               `json:"num_tree_per_iteration"`
	MaxFeatureIdx        int               `json:"max_feature_idx"`
	Trees                []json.RawMessage `json:"tree_info"`
	AverageOutput        bool              `json:"average_output"`
	Objective            string            `json:"objective"`
}

type lgbTreeJSON struct {
	NumLeaves int             `json:"num_leaves"`
	NumCat    uint32          `json:"num_cat"`
	RootRaw   json.RawMessage `json:"tree_structure"`
	Root      interface{}
}

type lgbNodeJSON struct {
	SplitIndex    uint32          `json:"split_index"`
	SplitFeature  uint32          `json:"split_feature"`
	Threshold     interface{}     `json:"threshold"`
	DecisionType  string          `json:"decision_type"`
	DefaultLeft   bool            `json:"default_left"`
	MissingType   string          `json:"missing_type"`
	LeftChildRaw  json.RawMessage `json:"left_child"`
	RightChildRaw json.RawMessage `json:"right_child"`
	LeftChild     interface{}
	RightChild    interface{}
}

var lgbStringToMissingType = map[string]uint8{
	"None": 0,
	"Zero": lgbFlagMissingZero,
	"NaN":  lgbFlagMissingNan,
}

// ParseLightGBMJSONFile 解析 LightGBM JSON 模型为 ForestIR。
func ParseLightGBMJSONFile(filename string) (*LightGBMLoadResult, error) {
	f, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return ParseLightGBMJSON(f)
}

// ParseLightGBMJSON 从 reader 解析 LightGBM JSON。
func ParseLightGBMJSON(r io.Reader) (*LightGBMLoadResult, error) {
	data := &lgbEnsembleJSON{}
	dec := json.NewDecoder(r)
	if err := dec.Decode(data); err != nil {
		return nil, err
	}
	if data.Name != "tree" {
		return nil, fmt.Errorf("expected 'name' field = 'tree' (got: '%s')", data.Name)
	}
	if data.Version != "v2" {
		return nil, fmt.Errorf("expected 'version' field = 'v2' (got: '%s')", data.Version)
	}
	if data.NumClasses != data.NumTreesPerIteration {
		return nil, fmt.Errorf(
			"meet case when num_class (%d) != num_tree_per_iteration (%d)",
			data.NumClasses, data.NumTreesPerIteration,
		)
	} else if data.NumClasses < 1 {
		return nil, fmt.Errorf("num_class (%d) should be > 0", data.NumClasses)
	} else if data.NumTreesPerIteration < 1 {
		return nil, fmt.Errorf("num_tree_per_iteration (%d) should be > 0", data.NumTreesPerIteration)
	}
	nTrees := len(data.Trees)
	if nTrees == 0 {
		return nil, fmt.Errorf("no trees in file")
	} else if nTrees%data.NumClasses != 0 {
		return nil, fmt.Errorf("wrong number of trees (%d) for number of class (%d)", nTrees, data.NumClasses)
	}
	name := "lightgbm.gbdt"
	if data.AverageOutput {
		name = "lightgbm.rf"
	}
	trees := make([]tree.TreeIR, 0, nTrees)
	for i := 0; i < nTrees; i++ {
		pt, err := lgbUnmarshalTree(data.Trees[i])
		if err != nil {
			return nil, fmt.Errorf("error while reading %d tree: %s", i, err.Error())
		}
		trees = append(trees, *pt.toIR())
	}
	nFeat := 0
	if data.MaxFeatureIdx > 0 {
		nFeat = data.MaxFeatureIdx + 1
	}
	obj := ""
	if !data.AverageOutput {
		obj = lgbObjectiveName(data.Objective)
	}
	return lgbResult(name, nFeat, data.NumClasses, trees, data.AverageOutput, obj), nil
}

func lgbUnmarshalNode(raw []byte) (interface{}, error) {
	node := &lgbNodeJSON{}
	if err := json.Unmarshal(raw, node); err != nil {
		return nil, err
	}
	if node.MissingType == "" {
		data := make(map[string]interface{})
		if err := json.Unmarshal(raw, &data); err != nil {
			return nil, err
		}
		value, ok := data["leaf_value"].(float64)
		if !ok {
			return nil, fmt.Errorf("unknown tree")
		}
		return value, nil
	}
	var err error
	node.LeftChild, err = lgbUnmarshalNode(node.LeftChildRaw)
	if err != nil {
		return nil, err
	}
	node.RightChild, err = lgbUnmarshalNode(node.RightChildRaw)
	if err != nil {
		return nil, err
	}
	return node, nil
}

func lgbUnmarshalTree(raw []byte) (lgbParsedTree, error) {
	t := lgbParsedTree{}
	treeJSON := &lgbTreeJSON{}
	if err := json.Unmarshal(raw, treeJSON); err != nil {
		return t, err
	}
	t.nCategorical = treeJSON.NumCat
	if t.nCategorical > 0 {
		t.catBoundaries = make([]uint32, 1)
	}
	if treeJSON.NumLeaves < 1 {
		return t, fmt.Errorf("num_leaves < 1")
	}
	numNodes := treeJSON.NumLeaves - 1
	var err error
	treeJSON.Root, err = lgbUnmarshalNode(treeJSON.RootRaw)
	if err != nil {
		return t, err
	}
	if value, ok := treeJSON.Root.(float64); ok {
		t.leafValues = append(t.leafValues, value)
		return t, nil
	}

	createNumericalNode := func(nodeJSON *lgbNodeJSON) (tree.LgNodeData, error) {
		var node tree.LgNodeData
		missingType, ok := lgbStringToMissingType[nodeJSON.MissingType]
		if !ok {
			return node, fmt.Errorf("unknown missing_type '%s'", nodeJSON.MissingType)
		}
		defaultType := uint8(0)
		if nodeJSON.DefaultLeft {
			defaultType = lgbFlagDefaultLeft
		}
		threshold, ok := nodeJSON.Threshold.(float64)
		if !ok {
			return node, fmt.Errorf("unexpected Threshold type %T", nodeJSON.Threshold)
		}
		node.Feature = nodeJSON.SplitFeature
		node.Flags = missingType | defaultType
		node.Threshold = threshold
		if value, ok := nodeJSON.LeftChild.(float64); ok {
			node.Flags |= lgbFlagLeftLeaf
			node.Left = uint32(len(t.leafValues))
			t.leafValues = append(t.leafValues, value)
		}
		if value, ok := nodeJSON.RightChild.(float64); ok {
			node.Flags |= lgbFlagRightLeaf
			node.Right = uint32(len(t.leafValues))
			t.leafValues = append(t.leafValues, value)
		}
		return node, nil
	}
	createCategoricalNode := func(nodeJSON *lgbNodeJSON) (tree.LgNodeData, error) {
		var node tree.LgNodeData
		missingType, ok := lgbStringToMissingType[nodeJSON.MissingType]
		if !ok {
			return node, fmt.Errorf("unknown missing_type '%s'", nodeJSON.MissingType)
		}
		thresholdString, ok := nodeJSON.Threshold.(string)
		if !ok {
			return node, fmt.Errorf("unexpected Threshold type %T", nodeJSON.Threshold)
		}
		tokens := strings.Split(thresholdString, "||")
		nBits := len(tokens)
		catIdx := uint32(0)
		catType := uint8(0)
		if nBits == 0 {
			return node, fmt.Errorf("no bits set")
		} else if nBits == 1 {
			value, err := strconv.Atoi(tokens[0])
			if err != nil {
				return node, fmt.Errorf("can't convert %s: %s", tokens[0], err.Error())
			}
			catIdx = uint32(value)
			catType = lgbFlagCatOneHot
		} else {
			thresholdValues := make([]int, len(tokens))
			for i, valueStr := range tokens {
				value, err := strconv.Atoi(valueStr)
				if err != nil {
					return node, fmt.Errorf("can't convert %s: %s", valueStr, err.Error())
				}
				thresholdValues[i] = value
			}
			bitset := util.ConstructBitset(thresholdValues)
			if len(bitset) == 1 {
				catIdx = bitset[0]
				catType = lgbFlagCatSmall
			} else {
				catIdx = uint32(len(t.catBoundaries) - 1)
				t.catThresholds = append(t.catThresholds, bitset...)
				t.catBoundaries = append(t.catBoundaries, uint32(len(t.catThresholds)))
			}
		}
		node.Feature = nodeJSON.SplitFeature
		node.Flags = lgbFlagCategorical | missingType | catType
		node.Threshold = float64(catIdx)
		if value, ok := nodeJSON.LeftChild.(float64); ok {
			node.Flags |= lgbFlagLeftLeaf
			node.Left = uint32(len(t.leafValues))
			t.leafValues = append(t.leafValues, value)
		}
		if value, ok := nodeJSON.RightChild.(float64); ok {
			node.Flags |= lgbFlagRightLeaf
			node.Right = uint32(len(t.leafValues))
			t.leafValues = append(t.leafValues, value)
		}
		return node, nil
	}
	createNode := func(nodeJSON *lgbNodeJSON) (tree.LgNodeData, error) {
		switch nodeJSON.DecisionType {
		case "==":
			return createCategoricalNode(nodeJSON)
		case "<=":
			return createNumericalNode(nodeJSON)
		default:
			return tree.LgNodeData{}, fmt.Errorf("unknown decision type '%s'", nodeJSON.DecisionType)
		}
	}

	type stackData struct {
		parentPtr *uint32
		nodeJSON  *lgbNodeJSON
	}
	stack := make([]stackData, 0, numNodes)
	root, ok := treeJSON.Root.(*lgbNodeJSON)
	if !ok {
		return t, fmt.Errorf("unexpected type of Root: %T", treeJSON.Root)
	}
	stack = append(stack, stackData{nil, root})
	t.nodes = make([]tree.LgNodeData, 0, numNodes)
	for len(stack) > 0 {
		sd := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		node, err := createNode(sd.nodeJSON)
		if err != nil {
			return t, err
		}
		if sd.parentPtr != nil {
			*sd.parentPtr = uint32(len(t.nodes))
		}
		t.nodes = append(t.nodes, node)
		if node.Flags&lgbFlagLeftLeaf == 0 {
			if left, ok := sd.nodeJSON.LeftChild.(*lgbNodeJSON); ok {
				stack = append(stack, stackData{&t.nodes[len(t.nodes)-1].Left, left})
			} else if _, ok := sd.nodeJSON.LeftChild.(float64); ok {
			} else {
				return t, fmt.Errorf("unexpected left child type %T", sd.nodeJSON.LeftChild)
			}
		}
		if node.Flags&lgbFlagRightLeaf == 0 {
			if right, ok := sd.nodeJSON.RightChild.(*lgbNodeJSON); ok {
				stack = append(stack, stackData{&t.nodes[len(t.nodes)-1].Right, right})
			} else if _, ok := sd.nodeJSON.RightChild.(float64); ok {
			} else {
				return t, fmt.Errorf("unexpected right child type %T", sd.nodeJSON.RightChild)
			}
		}
	}
	return t, nil
}
