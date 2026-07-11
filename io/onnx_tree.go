package io

import (
	"fmt"
	"strings"

	"github.com/linkerlin/leaves/tree"
)

// treeEnsembleAttrs 从 ONNX ai.onnx.ml TreeEnsembleRegressor 抽取的字段。
type treeEnsembleAttrs struct {
	nodesTreeids      []int64
	nodesNodeids      []int64
	nodesFeatureids   []int64
	nodesValues       []float64
	nodesModes        []string
	nodesTruenodeids  []int64
	nodesFalsenodeids []int64
	targetTreeids     []int64
	targetNodeids     []int64
	targetIds         []int64
	targetWeights     []float64
	nTargets          int
	baseValues        []float64
	aggregate         string
	postTransform     string
}

type onnxNodeRec struct {
	id, feat, trueID, falseID int
	value                     float64
	mode                      string
}

type onnxTKey struct{ tree, node int }

func attrsFromNode(n onnxNode) (treeEnsembleAttrs, error) {
	var a treeEnsembleAttrs
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
	a.targetTreeids = getInts("target_treeids")
	a.targetNodeids = getInts("target_nodeids")
	a.targetIds = getInts("target_ids")
	a.targetWeights = getFloats("target_weights")
	if x, ok := n.attrs["n_targets"]; ok {
		a.nTargets = int(x.i)
	}
	if a.nTargets <= 0 {
		a.nTargets = 1
	}
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
		return a, fmt.Errorf("onnx TreeEnsemble: empty nodes")
	}
	if len(a.nodesNodeids) != nn || len(a.nodesModes) != nn ||
		len(a.nodesFeatureids) != nn || len(a.nodesValues) != nn ||
		len(a.nodesTruenodeids) != nn || len(a.nodesFalsenodeids) != nn {
		return a, fmt.Errorf("onnx TreeEnsemble: nodes_* length mismatch")
	}
	if len(a.targetTreeids) == 0 || len(a.targetWeights) != len(a.targetTreeids) {
		return a, fmt.Errorf("onnx TreeEnsemble: target_* incomplete")
	}
	return a, nil
}

// forestFromTreeEnsemble 转为 ForestIR。
// 支持：BRANCH_LEQ + LEAF、SUM、post_transform=NONE、连续特征、标量叶（每树一个 class）。
func forestFromTreeEnsemble(a treeEnsembleAttrs) (*tree.ForestIR, error) {
	agg := strings.ToUpper(a.aggregate)
	if agg != "SUM" && agg != "0" {
		return nil, fmt.Errorf("onnx TreeEnsemble: only SUM aggregate supported, got %q", a.aggregate)
	}
	pt := strings.ToUpper(a.postTransform)
	if pt != "NONE" && pt != "0" && pt != "" {
		return nil, fmt.Errorf("onnx TreeEnsemble: only post_transform=NONE supported, got %q", a.postTransform)
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
	for i := range a.targetTreeids {
		tid := int(a.targetTreeids[i])
		nid := int(a.targetNodeids[i])
		oid := 0
		if i < len(a.targetIds) {
			oid = int(a.targetIds[i])
		}
		k := onnxTKey{tid, nid}
		if leafW[k] == nil {
			leafW[k] = map[int]float64{}
		}
		leafW[k][oid] = a.targetWeights[i]
	}

	nTargets := a.nTargets
	var treeIDs []int
	for tid := range byTree {
		treeIDs = append(treeIDs, tid)
	}
	// sort ascending
	for i := 0; i < len(treeIDs); i++ {
		for j := i + 1; j < len(treeIDs); j++ {
			if treeIDs[j] < treeIDs[i] {
				treeIDs[i], treeIDs[j] = treeIDs[j], treeIDs[i]
			}
		}
	}

	forest := &tree.ForestIR{
		NumFeatures:     maxFeat,
		NumOutputGroups: nTargets,
		Name:            "onnx.tree_ensemble",
		IterationIndptr: []int{0},
		BaseScore:       0,
	}
	if len(a.baseValues) > 0 {
		forest.BaseScore = a.baseValues[0]
		if len(a.baseValues) == nTargets {
			forest.BaseScores = append([]float64(nil), a.baseValues...)
		}
	}

	for _, tid := range treeIDs {
		tir, classIdx, err := buildONNXTreeIR(byTree[tid], leafW, tid, nTargets)
		if err != nil {
			return nil, fmt.Errorf("onnx tree %d: %w", tid, err)
		}
		forest.Trees = append(forest.Trees, *tir)
		forest.WeightDrop = append(forest.WeightDrop, 1)
		forest.TreeInfo = append(forest.TreeInfo, classIdx)
	}
	if nTargets > 1 && len(forest.Trees)%nTargets == 0 {
		for r := 1; r <= len(forest.Trees)/nTargets; r++ {
			forest.IterationIndptr = append(forest.IterationIndptr, r*nTargets)
		}
	} else {
		for i := 1; i <= len(forest.Trees); i++ {
			forest.IterationIndptr = append(forest.IterationIndptr, i)
		}
	}
	return forest, nil
}

func buildONNXTreeIR(nodes []onnxNodeRec, leafW map[onnxTKey]map[int]float64, tid, nTargets int) (*tree.TreeIR, int, error) {
	if len(nodes) == 0 {
		return nil, 0, fmt.Errorf("empty nodes")
	}
	byID := map[int]onnxNodeRec{}
	rootID := nodes[0].id
	minID := nodes[0].id
	for _, n := range nodes {
		byID[n.id] = n
		if n.id < minID {
			minID = n.id
			rootID = n.id
		}
	}
	// class: 主 target id（标量叶取第一个 weight 的 target）
	classIdx := 0
	// 收集叶子权重（按 node id）
	leafVal := map[int]float64{}
	for id, n := range byID {
		if !strings.Contains(n.mode, "LEAF") {
			continue
		}
		wmap := leafW[onnxTKey{tid, id}]
		if len(wmap) == 0 {
			leafVal[id] = 0
			continue
		}
		// 取最小 target id 的权重作为标量叶
		bestT, bestV := -1, 0.0
		for t, v := range wmap {
			if bestT < 0 || t < bestT {
				bestT, bestV = t, v
			}
		}
		leafVal[id] = bestV
		if bestT >= 0 {
			classIdx = bestT
		}
	}

	// 仅 BRANCH 节点进入 TreeIR 非叶表
	var branches []onnxNodeRec
	for _, n := range nodes {
		if strings.Contains(n.mode, "LEAF") {
			continue
		}
		if !strings.Contains(n.mode, "BRANCH") && !strings.Contains(n.mode, "LEQ") {
			// 未知 mode 当分支尝试
		}
		branches = append(branches, n)
	}

	// 单叶树
	if len(branches) == 0 {
		v := 0.0
		if w, ok := leafVal[rootID]; ok {
			v = w
		}
		return &tree.TreeIR{
			NumLeaves: 1,
			NumNodes:  0,
			MaxDepth:  0,
			LeafValue: []float64{v},
			OutputDim: 1,
		}, classIdx, nil
	}

	// 映射 branch 本地下标 0..nb-1；叶子用 ^leafIdx
	branchIdx := map[int]int{}
	for i, b := range branches {
		branchIdx[b.id] = i
	}
	// 叶子编号
	leafIDs := make([]int, 0)
	leafIndex := map[int]int{}
	addLeaf := func(id int) int {
		if idx, ok := leafIndex[id]; ok {
			return idx
		}
		idx := len(leafIDs)
		leafIDs = append(leafIDs, id)
		leafIndex[id] = idx
		return idx
	}
	// 预扫所有 true/false 目标
	for _, b := range branches {
		if _, isB := branchIdx[b.trueID]; !isB {
			addLeaf(b.trueID)
		}
		if _, isB := branchIdx[b.falseID]; !isB {
			addLeaf(b.falseID)
		}
	}

	nb := len(branches)
	tir := &tree.TreeIR{
		NumNodes:       nb,
		NumLeaves:      len(leafIDs),
		SplitFeature:   make([]int32, nb),
		SplitThreshold: make([]float64, nb),
		DefaultLeft:    make([]bool, nb),
		MissingZero:    make([]bool, nb),
		MissingNan:     make([]bool, nb),
		LeftChild:      make([]int32, nb),
		RightChild:     make([]int32, nb),
		LeafValue:      make([]float64, len(leafIDs)),
		OutputDim:      1,
		MaxDepth:       nb, // 上界
	}
	for i, id := range leafIDs {
		tir.LeafValue[i] = leafVal[id]
	}

	childRef := func(id int) int32 {
		if bi, ok := branchIdx[id]; ok {
			return int32(bi)
		}
		li := leafIndex[id]
		return int32(^li) // 负叶子编码
	}

	// 根必须是 branch 0：重排使 root 在 index 0
	rootBranch := -1
	if bi, ok := branchIdx[rootID]; ok {
		rootBranch = bi
	} else {
		rootBranch = 0
	}
	if rootBranch != 0 {
		branches[0], branches[rootBranch] = branches[rootBranch], branches[0]
		branchIdx = map[int]int{}
		for i, b := range branches {
			branchIdx[b.id] = i
		}
	}

	for i, b := range branches {
		tir.SplitFeature[i] = int32(b.feat)
		tir.SplitThreshold[i] = b.value
		tir.DefaultLeft[i] = true
		tir.MissingNan[i] = true
		// BRANCH_LEQ: x <= thr → true/left
		tir.LeftChild[i] = childRef(b.trueID)
		tir.RightChild[i] = childRef(b.falseID)
	}
	_ = nTargets
	return tir, classIdx, nil
}
