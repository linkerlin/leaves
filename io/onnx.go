// Package io — ONNX TreeEnsemble 极小子集导入（LIB-10）。
//
// 支持范围（experimental）：
//   - 算子：ai.onnx.ml TreeEnsembleRegressor
//   - 分裂：BRANCH_LEQ + LEAF；聚合 SUM；post_transform NONE
//   - 连续特征；标量叶（每树一个输出组）
//
// 不支持：完整 Graph 任意算子、分类器后处理、CATEGORY 分裂、向量叶多目标单树等。
// 更复杂模型请外部转为 XGBoost JSON / leaves.json。
package io

import (
	"fmt"
	"os"
	"strings"

	"github.com/linkerlin/leaves/model"
	"github.com/linkerlin/leaves/tree"
)

// ErrONNXNotImplemented 保留：完全无法解析或不在子集内时仍可 errors.Is。
// 成功加载后不再返回此错误。
var ErrONNXNotImplemented = fmt.Errorf("io: onnx import not implemented; convert to xgb json or leaves.json first")

// LoadONNX 从 ONNX 文件加载 TreeEnsembleRegressor 子集为 model.Ensemble。
func LoadONNX(path string, opts *LoadOptions) (*model.Ensemble, error) {
	if opts == nil {
		opts = DefaultLoadOptions()
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	forest, err := ParseONNXTreeEnsemble(data)
	if err != nil {
		// 保持可操作 hint
		return nil, &LoadError{
			Path:   path,
			Format: FormatONNX,
			Level:  SupportExperimental,
			Op:     "load",
			Msg:    err.Error(),
			Hint:   "仅支持 TreeEnsembleRegressor（BRANCH_LEQ/SUM/NONE）；复杂图请先转 XGB JSON / leaves.json",
			cause:  err,
		}
	}
	ir := &model.ModelIR{
		Forest:           forest,
		NumFeatures:      forest.NumFeatures,
		NRawOutputGroups: forest.NumOutputGroups,
		NOutputGroups:    forest.NumOutputGroups,
		Name:             forest.Name,
		Kind:             model.KindGBTree,
	}
	transform := tree.ApplyTransformRaw
	outType := tree.TransformRaw
	if opts.LoadTransformation || opts.AutoTransform {
		// 子集无 objective 元数据，保持 raw margin
		transform = tree.ApplyTransformRaw
		outType = tree.TransformRaw
	}
	return model.NewEnsembleFromIRWithHint(ir, transform, outType, opts.Backend, opts.Workload)
}

// ParseONNXTreeEnsemble 从 ONNX ModelProto 字节解析森林。
func ParseONNXTreeEnsemble(data []byte) (*tree.ForestIR, error) {
	nodes, err := parseONNXModel(data)
	if err != nil {
		return nil, err
	}
	var te *onnxNode
	for i := range nodes {
		n := &nodes[i]
		op := n.opType
		if strings.EqualFold(op, "TreeEnsembleRegressor") ||
			strings.HasSuffix(op, "TreeEnsembleRegressor") {
			te = n
			break
		}
	}
	if te == nil {
		return nil, fmt.Errorf("%w: no TreeEnsembleRegressor node (subset only)", ErrONNXNotImplemented)
	}
	attrs, err := attrsFromNode(*te)
	if err != nil {
		return nil, err
	}
	return forestFromTreeEnsemble(attrs)
}

// writeONNXTreeEnsembleRegressor 构造最小 ONNX 模型（测试/样例用）。
func writeONNXTreeEnsembleRegressor(attrs treeEnsembleAttrs) []byte {
	// NodeProto attributes
	var nodeBody []byte
	addFloats := func(name string, fs []float64) {
		var ab []byte
		ab = pbAppendString(ab, 1, name)
		ab = pbAppendInt64(ab, 20, 6) // FLOATS
		for _, f := range fs {
			ab = pbAppendFloat32(ab, 10, float32(f))
		}
		nodeBody = pbAppendBytes(nodeBody, 6, ab)
	}
	addInts := func(name string, xs []int64) {
		var ab []byte
		ab = pbAppendString(ab, 1, name)
		ab = pbAppendInt64(ab, 20, 7) // INTS
		for _, x := range xs {
			ab = pbAppendInt64(ab, 11, x)
		}
		nodeBody = pbAppendBytes(nodeBody, 6, ab)
	}
	addStrings := func(name string, ss []string) {
		var ab []byte
		ab = pbAppendString(ab, 1, name)
		ab = pbAppendInt64(ab, 20, 8) // STRINGS
		for _, s := range ss {
			ab = pbAppendString(ab, 12, s)
		}
		nodeBody = pbAppendBytes(nodeBody, 6, ab)
	}
	addInt := func(name string, v int64) {
		var ab []byte
		ab = pbAppendString(ab, 1, name)
		ab = pbAppendInt64(ab, 20, 2) // INT
		ab = pbAppendInt64(ab, 8, v)
		nodeBody = pbAppendBytes(nodeBody, 6, ab)
	}

	addInts("nodes_treeids", attrs.nodesTreeids)
	addInts("nodes_nodeids", attrs.nodesNodeids)
	addInts("nodes_featureids", attrs.nodesFeatureids)
	addFloats("nodes_values", attrs.nodesValues)
	addStrings("nodes_modes", attrs.nodesModes)
	addInts("nodes_truenodeids", attrs.nodesTruenodeids)
	addInts("nodes_falsenodeids", attrs.nodesFalsenodeids)
	addInts("target_treeids", attrs.targetTreeids)
	addInts("target_nodeids", attrs.targetNodeids)
	addInts("target_ids", attrs.targetIds)
	addFloats("target_weights", attrs.targetWeights)
	addInt("n_targets", int64(attrs.nTargets))
	if len(attrs.baseValues) > 0 {
		addFloats("base_values", attrs.baseValues)
	}
	addStrings("aggregate_function", []string{"SUM"})
	addStrings("post_transform", []string{"NONE"})

	// op_type + domain
	nodeBody = pbAppendString(nodeBody, 4, "TreeEnsembleRegressor")
	nodeBody = pbAppendString(nodeBody, 5, "ai.onnx.ml")
	// inputs/outputs minimal
	nodeBody = pbAppendString(nodeBody, 1, "X")
	nodeBody = pbAppendString(nodeBody, 2, "Y")

	var graph []byte
	graph = pbAppendBytes(graph, 1, nodeBody) // node

	var model []byte
	model = pbAppendInt64(model, 1, 8) // ir_version
	model = pbAppendBytes(model, 7, graph)
	return model
}

// SampleONNXStump 生成单 stump 测试模型：x0<=0.5 → 1.0 else 2.0。
func SampleONNXStump() []byte {
	return writeONNXTreeEnsembleRegressor(treeEnsembleAttrs{
		nodesTreeids:      []int64{0, 0, 0},
		nodesNodeids:      []int64{0, 1, 2},
		nodesFeatureids:   []int64{0, 0, 0},
		nodesValues:       []float64{0.5, 0, 0},
		nodesModes:        []string{"BRANCH_LEQ", "LEAF", "LEAF"},
		nodesTruenodeids:  []int64{1, 0, 0},
		nodesFalsenodeids: []int64{2, 0, 0},
		targetTreeids:     []int64{0, 0},
		targetNodeids:     []int64{1, 2},
		targetIds:         []int64{0, 0},
		targetWeights:     []float64{1.0, 2.0},
		nTargets:          1,
		baseValues:        []float64{0},
	})
}
