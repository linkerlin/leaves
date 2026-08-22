package io

// onnx_classifier.go — ai.onnx.ml TreeEnsembleClassifier 子集（ONNX-2）。
//
// 支持：BRANCH_LEQ + LEAF、aggregate SUM/AVERAGE、post_transform
// NONE / SOFTMAX（n 类，输出 n 组概率）/ LOGISTIC（二类，输出单组概率）。
// 不支持：CATEGORY 分裂、BRANCH_GT/LT 等其余分裂模式、多标签。

import (
	"fmt"
	"strings"

	"github.com/linkerlin/leaves/v2/tree"
)

// onnxOutTransform 分类器 post_transform → leaves 变换。
type onnxOutTransform struct {
	fn     tree.TransformFn
	outTyp tree.TransformType
}

// classifierAttrs 从 TreeEnsembleClassifier 节点抽取的字段。
type classifierAttrs struct {
	nodesTreeids      []int64
	nodesNodeids      []int64
	nodesFeatureids   []int64
	nodesValues       []float64
	nodesModes        []string
	nodesTruenodeids  []int64
	nodesFalsenodeids []int64
	classTreeids      []int64
	classNodeids      []int64
	classIds          []int64
	classWeights      []float64
	classLabelsInts   []int64
	classLabelsStrs   []string
	aggregate         string
	postTransform     string
	baseValues        []float64
}

func classifierAttrsFromNode(n onnxNode) (classifierAttrs, error) {
	var a classifierAttrs
	getInts := func(name string) []int64 {
		if x, ok := n.attrs[name]; ok {
			return x.ints
		}
		return nil
	}
	getFloats := func(name string) []float64 {
		if x, ok := n.attrs[name]; ok {
			return x.floats
		}
		return nil
	}
	getStrings := func(name string) []string {
		if x, ok := n.attrs[name]; ok {
			return x.strings
		}
		return nil
	}
	a.nodesTreeids = getInts("nodes_treeids")
	a.nodesNodeids = getInts("nodes_nodeids")
	a.nodesFeatureids = getInts("nodes_featureids")
	a.nodesValues = getFloats("nodes_values")
	a.nodesModes = getStrings("nodes_modes")
	a.nodesTruenodeids = getInts("nodes_truenodeids")
	a.nodesFalsenodeids = getInts("nodes_falsenodeids")
	a.classTreeids = getInts("class_treeids")
	a.classNodeids = getInts("class_nodeids")
	a.classIds = getInts("class_ids")
	a.classWeights = getFloats("class_weights")
	a.classLabelsInts = getInts("classlabels_int64s")
	if a.classLabelsInts == nil {
		a.classLabelsInts = getInts("classlabels_ints")
	}
	a.classLabelsStrs = getStrings("classlabels_strings")
	a.baseValues = getFloats("base_values")
	if x, ok := n.attrs["aggregate_function"]; ok && len(x.strings) > 0 {
		a.aggregate = string(x.strings[0])
	}
	if a.aggregate == "" {
		a.aggregate = "SUM"
	}
	if x, ok := n.attrs["post_transform"]; ok && len(x.strings) > 0 {
		a.postTransform = string(x.strings[0])
	}
	if a.postTransform == "" {
		a.postTransform = "NONE"
	}

	nn := len(a.nodesTreeids)
	if nn == 0 {
		return a, fmt.Errorf("onnx TreeEnsembleClassifier: empty nodes")
	}
	if len(a.nodesNodeids) != nn || len(a.nodesModes) != nn ||
		len(a.nodesFeatureids) != nn || len(a.nodesValues) != nn ||
		len(a.nodesTruenodeids) != nn || len(a.nodesFalsenodeids) != nn {
		return a, fmt.Errorf("onnx TreeEnsembleClassifier: nodes_* length mismatch")
	}
	if len(a.classTreeids) == 0 || len(a.classWeights) != len(a.classTreeids) {
		return a, fmt.Errorf("onnx TreeEnsembleClassifier: class_* incomplete")
	}
	return a, nil
}

// numClasses 类数：优先 classlabels；否则按 max(class_id)+1 推断。
func (a classifierAttrs) numClasses() int {
	if n := len(a.classLabelsInts); n > 0 {
		return n
	}
	if n := len(a.classLabelsStrs); n > 0 {
		return n
	}
	max := -1
	for _, id := range a.classIds {
		if int(id) > max {
			max = int(id)
		}
	}
	return max + 1
}

// forestFromClassifier 转 ForestIR + 输出变换。
// 每树归属单一类（其叶子 class_id）；AVERAGE 时按类树数折算权重；
// LOGISTIC 仅二类合法且坍缩为单输出组（class 1 的 raw → sigmoid）。
func forestFromClassifier(a classifierAttrs) (*tree.ForestIR, onnxOutTransform, error) {
	agg := strings.ToUpper(a.aggregate)
	if agg != "SUM" && agg != "AVERAGE" {
		return nil, onnxOutTransform{}, fmt.Errorf("onnx TreeEnsembleClassifier: aggregate %q unsupported (SUM/AVERAGE)", a.aggregate)
	}
	pt := strings.ToUpper(strings.TrimSpace(a.postTransform))
	nClasses := a.numClasses()
	if nClasses < 2 {
		return nil, onnxOutTransform{}, fmt.Errorf("onnx TreeEnsembleClassifier: need >=2 classes, got %d", nClasses)
	}

	out := onnxOutTransform{fn: tree.ApplyTransformRaw, outTyp: tree.TransformRaw}
	binaryLogistic := false
	switch pt {
	case "NONE", "":
	case "SOFTMAX":
		out.fn, out.outTyp = tree.ApplyTransformSoftmax, tree.TransformSoftmax
	case "LOGISTIC":
		if nClasses != 2 {
			return nil, onnxOutTransform{}, fmt.Errorf("onnx TreeEnsembleClassifier: LOGISTIC requires 2 classes, got %d", nClasses)
		}
		out.fn, out.outTyp = tree.ApplyTransformLogistic, tree.TransformLogistic
		binaryLogistic = true
	default:
		return nil, onnxOutTransform{}, fmt.Errorf("onnx TreeEnsembleClassifier: post_transform %q unsupported (NONE/SOFTMAX/LOGISTIC)", a.postTransform)
	}

	byTree := map[int][]onnxNodeRec{}
	maxFeat := 0
	for i := range a.nodesTreeids {
		tid := int(a.nodesTreeids[i])
		rec := onnxNodeRec{
			id:      int(a.nodesNodeids[i]),
			feat:    int(a.nodesFeatureids[i]),
			trueID:  int(a.nodesTruenodeids[i]),
			falseID: int(a.nodesFalsenodeids[i]),
			value:   a.nodesValues[i],
			mode:    strings.ToUpper(strings.TrimSpace(string(a.nodesModes[i]))),
		}
		if rec.feat >= 0 && rec.feat+1 > maxFeat {
			maxFeat = rec.feat + 1
		}
		byTree[tid] = append(byTree[tid], rec)
	}

	leafW := map[onnxTKey]map[int]float64{}
	treesPerClass := map[int]int{}
	for i := range a.classTreeids {
		tid := int(a.classTreeids[i])
		nid := int(a.classNodeids[i])
		cid := 0
		if i < len(a.classIds) {
			cid = int(a.classIds[i])
		}
		k := onnxTKey{tid, nid}
		if leafW[k] == nil {
			leafW[k] = map[int]float64{}
		}
		leafW[k][cid] = a.classWeights[i]
	}
	// 类归属：每树取其任一叶子的 class_id（转换器产出单类树）。
	treeClass := map[int]int{}
	for k, wmap := range leafW {
		for cid := range wmap {
			treeClass[k.tree] = cid
			break
		}
	}
	for _, c := range treeClass {
		treesPerClass[c]++
	}

	var treeIDs []int
	for tid := range byTree {
		treeIDs = append(treeIDs, tid)
	}
	for i := 0; i < len(treeIDs); i++ {
		for j := i + 1; j < len(treeIDs); j++ {
			if treeIDs[j] < treeIDs[i] {
				treeIDs[i], treeIDs[j] = treeIDs[j], treeIDs[i]
			}
		}
	}

	nGroups := nClasses
	if binaryLogistic {
		nGroups = 1
	}
	forest := &tree.ForestIR{
		NumFeatures:     maxFeat,
		NumOutputGroups: nGroups,
		Name:            "onnx.tree_ensemble_classifier",
		IterationIndptr: []int{0},
	}
	if len(a.baseValues) == nClasses {
		forest.BaseScores = append([]float64(nil), a.baseValues...)
		forest.BaseScore = a.baseValues[0]
		if binaryLogistic {
			forest.BaseScore = a.baseValues[1]
			forest.BaseScores = []float64{a.baseValues[1]}
		}
	} else if len(a.baseValues) == 1 {
		forest.BaseScore = a.baseValues[0]
	}

	for _, tid := range treeIDs {
		classIdx := treeClass[tid]
		if binaryLogistic && classIdx == 0 {
			// 二类 LOGISTIC：仅 class-1 树进森林（p = sigmoid(raw)），
			// 且组号坍缩到 0（NOutputGroups=1）。
			continue
		}
		tir, _, err := buildONNXTreeIR(byTree[tid], leafW, tid, 1)
		if err != nil {
			return nil, onnxOutTransform{}, fmt.Errorf("onnx classifier tree %d: %w", tid, err)
		}
		if agg == "AVERAGE" {
			d := float64(treesPerClass[classIdx])
			if d > 0 {
				for i := range tir.LeafValue {
					tir.LeafValue[i] /= d
				}
			}
		}
		info := classIdx
		if binaryLogistic {
			info = 0
		}
		forest.Trees = append(forest.Trees, *tir)
		forest.WeightDrop = append(forest.WeightDrop, 1)
		forest.TreeInfo = append(forest.TreeInfo, info)
		forest.IterationIndptr = append(forest.IterationIndptr, len(forest.Trees))
	}
	return forest, out, nil
}
