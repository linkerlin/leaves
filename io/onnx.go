// Package io — ONNX 模型导入。
//
// 两条路径：
//   - LoadONNX：ai.onnx.ml TreeEnsemble 子集（LIB-10 + ONNX-2）：
//     Regressor（BRANCH_LEQ/SUM/NONE）与 Classifier（SUM/AVERAGE ×
//     NONE/SOFTMAX/LOGISTIC-二类），转 ForestIR 走 Native/Born 树引擎；wasm 可用。
//   - LoadOnnxGraph：完整 ONNX 计算图（任意算子，30+，opset 1–21）via born 运行时；
//     通用 NN/图推理；返回 OnnxModel；仅非 wasm。
//
// 仍不支持：CATEGORY 分裂、向量叶多目标单树（TreeEnsemble 子集内）。
package io

import (
	"fmt"
	"os"
	"strings"

	"github.com/linkerlin/leaves/v2/model"
	"github.com/linkerlin/leaves/v2/tree"
)

// ErrONNXNotImplemented 保留：完全无法解析或不在子集内时仍可 errors.Is。
// 成功加载后不再返回此错误。
var ErrONNXNotImplemented = fmt.Errorf("io: onnx import not implemented; convert to xgb json or leaves.json first")

// LoadONNX 从 ONNX 文件加载 TreeEnsemble 子集（Regressor 或 Classifier）为 model.Ensemble。
func LoadONNX(path string, opts *LoadOptions) (*model.Ensemble, error) {
	if opts == nil {
		opts = DefaultLoadOptions()
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	ir, outT, err := parseONNXForest(data)
	if err != nil {
		// 保持可操作 hint
		return nil, &LoadError{
			Path:   path,
			Format: FormatONNX,
			Level:  SupportExperimental,
			Op:     "load",
			Msg:    err.Error(),
			Hint:   "TreeEnsemble 子集：Regressor（BRANCH_LEQ/SUM/NONE）或 Classifier（SUM/AVERAGE × NONE/SOFTMAX/LOGISTIC-二类）；通用图用 LoadOnnxGraph（非 wasm）或先转 XGB JSON / leaves.json",
			cause:  err,
		}
	}
	// 子集无 objective 元数据；Classifier 的 post_transform 即变换真源，
	// 用户 LoadTransformation/AutoTransform 不覆盖 ONNX 声明的后处理。
	transform, outType := outT.fn, outT.outTyp
	return model.NewEnsembleFromIRWithHint(ir, transform, outType, opts.Backend, opts.Workload)
}

// parseONNXForest 识别 TreeEnsembleRegressor / TreeEnsembleClassifier 并转 IR。
func parseONNXForest(data []byte) (*model.ModelIR, onnxOutTransform, error) {
	nodes, err := parseONNXModel(data)
	if err != nil {
		return nil, onnxOutTransform{}, err
	}
	raw := onnxOutTransform{fn: tree.ApplyTransformRaw, outTyp: tree.TransformRaw}
	var te, cls *onnxNode
	for i := range nodes {
		n := &nodes[i]
		op := n.opType
		if strings.EqualFold(op, "TreeEnsembleRegressor") || strings.HasSuffix(op, "TreeEnsembleRegressor") {
			te = n
		}
		if strings.EqualFold(op, "TreeEnsembleClassifier") || strings.HasSuffix(op, "TreeEnsembleClassifier") {
			cls = n
		}
	}
	if cls != nil {
		attrs, err := classifierAttrsFromNode(*cls)
		if err != nil {
			return nil, onnxOutTransform{}, err
		}
		forest, outT, err := forestFromClassifier(attrs)
		if err != nil {
			return nil, onnxOutTransform{}, err
		}
		return &model.ModelIR{
			Forest:           forest,
			NumFeatures:      forest.NumFeatures,
			NRawOutputGroups: forest.NumOutputGroups,
			NOutputGroups:    forest.NumOutputGroups,
			Name:             forest.Name,
			Kind:             model.KindGBTree,
		}, outT, nil
	}
	if te != nil {
		attrs, err := attrsFromNode(*te)
		if err != nil {
			return nil, onnxOutTransform{}, err
		}
		forest, err := forestFromTreeEnsemble(attrs)
		if err != nil {
			return nil, onnxOutTransform{}, err
		}
		return &model.ModelIR{
			Forest:           forest,
			NumFeatures:      forest.NumFeatures,
			NRawOutputGroups: forest.NumOutputGroups,
			NOutputGroups:    forest.NumOutputGroups,
			Name:             forest.Name,
			Kind:             model.KindGBTree,
		}, raw, nil
	}
	return nil, onnxOutTransform{}, fmt.Errorf("%w: no TreeEnsembleRegressor/Classifier node (subset only)", ErrONNXNotImplemented)
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

// —— AttributeProto 写入助手（编码与本包 parser 字段号一致；测试/样例共用）——

func attrFloats(name string, fs []float64) []byte {
	var ab []byte
	ab = pbAppendString(ab, 1, name)
	ab = pbAppendInt64(ab, 20, 6) // FLOATS
	for _, f := range fs {
		ab = pbAppendFloat32(ab, 10, float32(f))
	}
	return pbAppendBytes(nil, 6, ab)
}

func attrInts(name string, xs []int64) []byte {
	var ab []byte
	ab = pbAppendString(ab, 1, name)
	ab = pbAppendInt64(ab, 20, 7) // INTS
	for _, x := range xs {
		ab = pbAppendInt64(ab, 11, x)
	}
	return pbAppendBytes(nil, 6, ab)
}

func attrStrings(name string, ss []string) []byte {
	var ab []byte
	ab = pbAppendString(ab, 1, name)
	ab = pbAppendInt64(ab, 20, 8) // STRINGS
	for _, s := range ss {
		ab = pbAppendString(ab, 12, s)
	}
	return pbAppendBytes(nil, 6, ab)
}

func attrInt(name string, v int64) []byte {
	var ab []byte
	ab = pbAppendString(ab, 1, name)
	ab = pbAppendInt64(ab, 20, 2) // INT
	ab = pbAppendInt64(ab, 8, v)
	return pbAppendBytes(nil, 6, ab)
}

// writeONNXTreeEnsembleRegressor 构造最小 ONNX 模型（测试/样例用）。
func writeONNXTreeEnsembleRegressor(attrs treeEnsembleAttrs) []byte {
	nodeBody := attrInts("nodes_treeids", attrs.nodesTreeids)
	nodeBody = append(nodeBody, attrInts("nodes_nodeids", attrs.nodesNodeids)...)
	nodeBody = append(nodeBody, attrInts("nodes_featureids", attrs.nodesFeatureids)...)
	nodeBody = append(nodeBody, attrFloats("nodes_values", attrs.nodesValues)...)
	nodeBody = append(nodeBody, attrStrings("nodes_modes", attrs.nodesModes)...)
	nodeBody = append(nodeBody, attrInts("nodes_truenodeids", attrs.nodesTruenodeids)...)
	nodeBody = append(nodeBody, attrInts("nodes_falsenodeids", attrs.nodesFalsenodeids)...)
	nodeBody = append(nodeBody, attrInts("target_treeids", attrs.targetTreeids)...)
	nodeBody = append(nodeBody, attrInts("target_nodeids", attrs.targetNodeids)...)
	nodeBody = append(nodeBody, attrInts("target_ids", attrs.targetIds)...)
	nodeBody = append(nodeBody, attrFloats("target_weights", attrs.targetWeights)...)
	nodeBody = append(nodeBody, attrInt("n_targets", int64(attrs.nTargets))...)
	if len(attrs.baseValues) > 0 {
		nodeBody = append(nodeBody, attrFloats("base_values", attrs.baseValues)...)
	}
	nodeBody = append(nodeBody, attrStrings("aggregate_function", []string{"SUM"})...)
	nodeBody = append(nodeBody, attrStrings("post_transform", []string{"NONE"})...)

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

// WriteONNXTreeEnsembleClassifier 构造最小分类器 ONNX 模型（测试/样例用）。
// 每类一 stump（x0<=0.5 → true 叶）；classWeights 与 class_* 等长。
func WriteONNXTreeEnsembleClassifier(
	nodesTreeids, nodesNodeids, nodesFeatureids, nodesTruenodeids, nodesFalsenodeids []int64,
	nodesValues []float64, nodesModes []string,
	classTreeids, classNodeids, classIds []int64, classWeights []float64,
	classLabels []int64, aggregate, postTransform string,
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
	var model []byte
	model = pbAppendInt64(model, 1, 8)
	model = pbAppendBytes(model, 7, graph)
	return model
}

// SampleONNXClassifier3Class 生成三类分类器测试模型（SOFTMAX）：
// 每类一棵 stump（x0<=0.5）：class0 叶权 [2,0]，class1 [0,2]，class2 [1,1]。
func SampleONNXClassifier3Class() []byte {
	return WriteONNXTreeEnsembleClassifier(
		[]int64{0, 0, 0, 1, 1, 1, 2, 2, 2}, // nodes_treeids
		[]int64{0, 1, 2, 0, 1, 2, 0, 1, 2}, // nodes_nodeids
		[]int64{0, 0, 0, 0, 0, 0, 0, 0, 0}, // nodes_featureids
		[]int64{1, 0, 0, 1, 0, 0, 1, 0, 0}, // nodes_truenodeids
		[]int64{2, 0, 0, 2, 0, 0, 2, 0, 0}, // nodes_falsenodeids
		[]float64{0.5, 0, 0, 0.5, 0, 0, 0.5, 0, 0},
		[]string{"BRANCH_LEQ", "LEAF", "LEAF", "BRANCH_LEQ", "LEAF", "LEAF", "BRANCH_LEQ", "LEAF", "LEAF"},
		[]int64{0, 0, 1, 1, 2, 2},   // class_treeids
		[]int64{1, 2, 1, 2, 1, 2},   // class_nodeids
		[]int64{0, 0, 1, 1, 2, 2},   // class_ids
		[]float64{2, 0, 0, 2, 1, 1}, // class_weights
		[]int64{10, 20, 30},         // classlabels
		"SUM", "SOFTMAX",
	)
}

// SampleONNXClassifierBinary 生成二类 LOGISTIC 测试模型：
// class1 树（x0<=0.5 → 1.0 else -1.0）；p = sigmoid(raw)。
func SampleONNXClassifierBinary() []byte {
	return WriteONNXTreeEnsembleClassifier(
		[]int64{0, 0, 0},
		[]int64{0, 1, 2},
		[]int64{0, 0, 0},
		[]int64{1, 0, 0},
		[]int64{2, 0, 0},
		[]float64{0.5, 0, 0},
		[]string{"BRANCH_LEQ", "LEAF", "LEAF"},
		[]int64{0, 0},
		[]int64{1, 2},
		[]int64{1, 1},
		[]float64{1.0, -1.0},
		[]int64{0, 1},
		"SUM", "LOGISTIC",
	)
}
