package io

import (
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"strconv"
	"strings"

	"github.com/linkerlin/leaves/internal/xgbin"
	"github.com/linkerlin/leaves/linear"
	"github.com/linkerlin/leaves/model"
	"github.com/linkerlin/leaves/tree"
)

// XGBoostLoadResult JSON 加载结果。
type XGBoostLoadResult struct {
	IR        *model.ModelIR
	Objective string
}

// ParseXGBoostJSON 从 reader 解析 XGBoost 3.x JSON 模型。
func ParseXGBoostJSON(r io.Reader) (*XGBoostLoadResult, error) {
	data, err := io.ReadAll(r)
	if err != nil {
		return nil, err
	}
	return parseXGBoostJSONBytes(data)
}

// ParseXGBoostJSONFile 从文件解析 XGBoost JSON 模型。
func ParseXGBoostJSONFile(filename string) (*XGBoostLoadResult, error) {
	data, err := os.ReadFile(filename)
	if err != nil {
		return nil, err
	}
	return parseXGBoostJSONBytes(data)
}

func parseXGBoostJSONBytes(data []byte) (*XGBoostLoadResult, error) {
	var root map[string]json.RawMessage
	if err := json.Unmarshal(data, &root); err != nil {
		return nil, fmt.Errorf("invalid xgboost json: %w", err)
	}
	xgbVersion, _ := parseIntArray(root["version"])
	learnerRaw, ok := root["learner"]
	if !ok {
		return nil, fmt.Errorf("missing learner field")
	}
	var learner map[string]json.RawMessage
	if err := json.Unmarshal(learnerRaw, &learner); err != nil {
		return nil, err
	}

	lmp, err := parseLearnerModelParam(learner["learner_model_param"])
	if err != nil {
		return nil, err
	}
	boostFromAverage := lmp.boostFromAverage

	objName := ""
	if objRaw, ok := learner["objective"]; ok {
		var obj map[string]json.RawMessage
		if json.Unmarshal(objRaw, &obj) == nil {
			objName, _ = parseStringField(obj["name"])
		}
	}

	featureNames, _ := parseStringArrayField(learner["feature_names"])
	featureTypes, _ := parseStringArrayField(learner["feature_types"])

	gbRaw, ok := learner["gradient_booster"]
	if !ok {
		return nil, fmt.Errorf("missing gradient_booster")
	}
	var booster map[string]json.RawMessage
	if err := json.Unmarshal(gbRaw, &booster); err != nil {
		return nil, err
	}
	boosterName, _ := parseStringField(booster["name"])
	modelRaw, boosterWDRaw, err := resolveGBBoosterModel(booster)
	if err != nil {
		return nil, err
	}

	if boosterName == "gblinear" {
		return parseXGBoostJSONGblinear(modelRaw, lmp, featureNames, featureTypes, objName, xgbVersion, boostFromAverage)
	}

	forest, err := parseGBTreeModel(modelRaw, lmp, boosterName)
	if err != nil {
		return nil, err
	}
	if boosterName == "dart" && len(boosterWDRaw) > 0 {
		if wd, err := parseFloatArray(boosterWDRaw); err == nil && len(wd) == len(forest.Trees) {
			forest.WeightDrop = wd
		}
	}
	adjustBaseScoreForObjective(objName, forest)

	kind := model.KindGBTree
	name := "xgboost.gbtree"
	if boosterName == "dart" {
		kind = model.KindDART
		name = "xgboost.dart"
	}

	numClass := lmp.numClass
	if numClass <= 0 {
		numClass = lmp.numTarget
	}
	if numClass <= 0 {
		numClass = 1
	}

	return &XGBoostLoadResult{
		IR: &model.ModelIR{
			Kind:             kind,
			NumFeatures:      lmp.numFeatures,
			NRawOutputGroups: numClass,
			NOutputGroups:    numClass,
			Name:             name,
			Forest:           forest,
			FeatureNames:     featureNames,
			FeatureTypes:     featureTypes,
			XGBVersion:       xgbVersion,
			XGBBoostFromAverage: boostFromAverage,
		},
		Objective: objName,
	}, nil
}

func parseXGBoostJSONGblinear(
	modelRaw json.RawMessage,
	lmp learnerModelParam,
	featureNames, featureTypes []string,
	objName string,
	xgbVersion []int,
	boostFromAverage bool,
) (*XGBoostLoadResult, error) {
	lin, err := parseGBLinearModel(modelRaw, lmp)
	if err != nil {
		return nil, err
	}
	adjustLinearBaseScoreForObjective(objName, lin)

	numClass := lmp.numClass
	if numClass <= 0 {
		numClass = lmp.numTarget
	}
	if numClass <= 0 {
		numClass = lin.NumOutputGroups
	}
	if numClass <= 0 {
		numClass = 1
	}

	return &XGBoostLoadResult{
		IR: &model.ModelIR{
			Kind:                model.KindGBLinear,
			NumFeatures:         lin.NumFeatures,
			NRawOutputGroups:    numClass,
			NOutputGroups:       numClass,
			Name:                "xgboost.gblinear",
			Linear:              lin,
			FeatureNames:        featureNames,
			FeatureTypes:        featureTypes,
			XGBVersion:          xgbVersion,
			XGBBoostFromAverage: boostFromAverage,
		},
		Objective: objName,
	}, nil
}

func resolveGBBoosterModel(booster map[string]json.RawMessage) (json.RawMessage, json.RawMessage, error) {
	if raw, ok := booster["model"]; ok {
		return raw, booster["weight_drop"], nil
	}
	if gbRaw, ok := booster["gbtree"]; ok {
		var nested map[string]json.RawMessage
		if err := json.Unmarshal(gbRaw, &nested); err != nil {
			return nil, nil, err
		}
		if raw, ok := nested["model"]; ok {
			return raw, booster["weight_drop"], nil
		}
	}
	return nil, nil, fmt.Errorf("missing gradient_booster model")
}

func parseGBLinearModel(raw json.RawMessage, lmp learnerModelParam) (*linear.LinearIR, error) {
	var obj map[string]json.RawMessage
	if err := json.Unmarshal(raw, &obj); err != nil {
		return nil, err
	}
	weightsRaw, ok := obj["weights"]
	if !ok {
		return nil, fmt.Errorf("gblinear: missing weights")
	}
	weights, err := parseFloatArray(weightsRaw)
	if err != nil {
		return nil, fmt.Errorf("gblinear weights: %w", err)
	}

	nf := lmp.numFeatures
	if nf <= 0 {
		return nil, fmt.Errorf("gblinear: zero num_feature")
	}
	ng := lmp.numClass
	if ng <= 0 {
		ng = lmp.numTarget
	}
	if ng <= 0 {
		ng = inferGBLinearOutputGroups(nf, len(weights))
	}
	expected := (nf + 1) * ng
	if len(weights) != expected {
		return nil, fmt.Errorf("gblinear: weights len %d != (%d+1)*%d", len(weights), nf, ng)
	}

	return &linear.LinearIR{
		NumFeatures:     nf,
		NumOutputGroups: ng,
		BaseScore:       lmp.baseScore,
		Weights:         weights,
		Name:            "xgboost.gblinear",
	}, nil
}

func inferGBLinearOutputGroups(numFeatures, weightLen int) int {
	if numFeatures <= 0 || weightLen <= 0 {
		return 1
	}
	denom := numFeatures + 1
	if weightLen%denom != 0 {
		return 1
	}
	return weightLen / denom
}

func adjustLinearBaseScoreForObjective(objective string, lin *linear.LinearIR) {
	if lin == nil {
		return
	}
	switch objective {
	case "binary:logistic", "reg:logistic":
		lin.BaseScore = probabilityToLogit(lin.BaseScore)
	}
}

type learnerModelParam struct {
	baseScore         float64
	baseScores        []float64
	numFeatures       int
	numClass          int
	numTarget         int
	boostFromAverage  bool
}

func parseLearnerModelParam(raw json.RawMessage) (learnerModelParam, error) {
	var out learnerModelParam
	if len(raw) == 0 {
		return out, fmt.Errorf("missing learner_model_param")
	}
	var obj map[string]json.RawMessage
	if err := json.Unmarshal(raw, &obj); err != nil {
		return out, err
	}
	if v, ok := obj["num_feature"]; ok {
		n, err := parseIntField(v)
		if err != nil {
			return out, err
		}
		out.numFeatures = n
	}
	if v, ok := obj["num_class"]; ok {
		n, err := parseIntField(v)
		if err == nil {
			out.numClass = n
		}
	}
	if v, ok := obj["num_target"]; ok {
		n, err := parseIntField(v)
		if err == nil {
			out.numTarget = n
		}
	}
	if v, ok := obj["base_score"]; ok {
		bs, scores, err := parseBaseScore(v)
		if err != nil {
			return out, err
		}
		out.baseScore = bs
		out.baseScores = scores
	}
	if v, ok := obj["boost_from_average"]; ok {
		n, err := parseIntField(v)
		if err == nil {
			out.boostFromAverage = n != 0
		}
	}
	return out, nil
}

func parseGBTreeModel(raw json.RawMessage, lmp learnerModelParam, boosterName string) (*tree.ForestIR, error) {
	var obj map[string]json.RawMessage
	if err := json.Unmarshal(raw, &obj); err != nil {
		return nil, err
	}

	var numParallelTree int
	if gp, ok := obj["gbtree_model_param"]; ok {
		var gmp map[string]json.RawMessage
		if json.Unmarshal(gp, &gmp) == nil {
			if v, ok := gmp["num_parallel_tree"]; ok {
				numParallelTree, _ = parseIntField(v)
			}
		}
	}

	iterationIndptr, _ := parseIntArray(obj["iteration_indptr"])
	treeInfo, _ := parseIntArray(obj["tree_info"])

	treesRaw, ok := obj["trees"]
	if !ok {
		return nil, fmt.Errorf("missing trees array")
	}
	var trees []json.RawMessage
	if err := json.Unmarshal(treesRaw, &trees); err != nil {
		return nil, err
	}

	numFeatures := uint32(lmp.numFeatures)
	if numFeatures == 0 {
		return nil, fmt.Errorf("zero number of features")
	}

	numOutputGroups := lmp.numClass
	if numOutputGroups <= 0 {
		numOutputGroups = lmp.numTarget
	}
	if numOutputGroups <= 0 {
		numOutputGroups = 1
	}
	if len(treeInfo) == 0 {
		treeInfo = make([]int, len(trees))
	}

	forest := &tree.ForestIR{
		NumFeatures:       lmp.numFeatures,
		NumOutputGroups:   numOutputGroups,
		BaseScore:         lmp.baseScore,
		BaseScores:        lmp.baseScores,
		TreeInfo:          treeInfo,
		IterationIndptr:   iterationIndptr,
		NumParallelTree:   numParallelTree,
		Trees:             make([]tree.TreeIR, 0, len(trees)),
		AverageOutput:     false,
	}

	if boosterName == "dart" {
		forest.Name = "xgboost.dart"
	} else {
		forest.Name = "xgboost.gbtree"
	}

	weightDrop := make([]float64, len(trees))
	for i := range weightDrop {
		weightDrop[i] = 1.0
	}
	if wdRaw, ok := obj["weight_drop"]; ok {
		if wd, err := parseFloatArray(wdRaw); err == nil && len(wd) == len(trees) {
			weightDrop = wd
		}
	}
	forest.WeightDrop = weightDrop

	if catsRaw, ok := obj["cats"]; ok && len(catsRaw) > 0 {
		forest.XGBCatsRaw = append([]byte(nil), catsRaw...)
	}

	for i, tr := range trees {
		forest.XGBTreesRaw = append(forest.XGBTreesRaw, append([]byte(nil), tr...))
		tm, catMeta, err := parseXGBTreeJSON(tr)
		if err != nil {
			return nil, fmt.Errorf("tree %d: %w", i, err)
		}
		tir, err := tree.TreeIRFromXGBModel(tm, numFeatures)
		if err != nil {
			return nil, fmt.Errorf("tree %d: %w", i, err)
		}
		if catMeta != nil {
			tree.ApplyXGBCategoricalSplits(
				tir,
				catMeta.splitType,
				catMeta.categories,
				catMeta.categoriesNodes,
				catMeta.categoriesSegments,
				catMeta.categoriesSizes,
			)
		}
		if outputDim := tm.OutputDim; outputDim > 1 {
			tir.OutputDim = outputDim
		}
		forest.Trees = append(forest.Trees, *tir)
	}

	if len(iterationIndptr) == 0 {
		forest.IterationIndptr = make([]int, len(trees)/numOutputGroups+1)
		for i := range forest.IterationIndptr {
			forest.IterationIndptr[i] = i * numOutputGroups
		}
	}

	return forest, nil
}

func parseXGBTreeJSON(raw json.RawMessage) (*xgbin.TreeModel, *xgbTreeCatMeta, error) {
	var obj map[string]json.RawMessage
	if err := json.Unmarshal(raw, &obj); err != nil {
		return nil, nil, err
	}

	tpRaw, ok := obj["tree_param"]
	if !ok {
		return nil, nil, fmt.Errorf("missing tree_param")
	}
	var tp map[string]json.RawMessage
	if err := json.Unmarshal(tpRaw, &tp); err != nil {
		return nil, nil, err
	}
	numNodes, err := parseIntField(tp["num_nodes"])
	if err != nil {
		return nil, nil, err
	}
	numFeature, _ := parseIntField(tp["num_feature"])
	sizeLeafVector, _ := parseIntField(tp["size_leaf_vector"])
	if sizeLeafVector <= 0 {
		sizeLeafVector = 1
	}

	lefts, err := parseInt32Array(obj["left_children"])
	if err != nil {
		return nil, nil, err
	}
	rights, err := parseInt32Array(obj["right_children"])
	if err != nil {
		return nil, nil, err
	}
	indices, err := parseInt32Array(obj["split_indices"])
	if err != nil {
		return nil, nil, err
	}
	conds, err := parseFloatArray(obj["split_conditions"])
	if err != nil {
		return nil, nil, err
	}
	dftLeft, err := parseBoolArray(obj["default_left"])
	if err != nil {
		return nil, nil, err
	}

	if len(lefts) != numNodes {
		return nil, nil, fmt.Errorf("left_children size %d != num_nodes %d", len(lefts), numNodes)
	}

	nodes := make([]xgbin.Node, numNodes)
	stats := make([]xgbin.RTreeNodeStat, numNodes)
	if lossRaw, ok := obj["loss_changes"]; ok {
		if losses, err := parseFloatArray(lossRaw); err == nil && len(losses) == numNodes {
			for i := range stats {
				stats[i].LossChg = float32(losses[i])
			}
		}
	}
	if hessRaw, ok := obj["sum_hessian"]; ok {
		if hess, err := parseFloatArray(hessRaw); err == nil && len(hess) == numNodes {
			for i := range stats {
				stats[i].SumHess = float32(hess[i])
			}
		}
	}

	for i := 0; i < numNodes; i++ {
		nodes[i].CLeft = lefts[i]
		nodes[i].CRight = rights[i]
		nodes[i].Info = float32(conds[i])
		feat := uint32(0)
		if i < len(indices) {
			feat = uint32(indices[i])
		}
		if i < len(dftLeft) && dftLeft[i] {
			feat |= 1 << 31
		}
		nodes[i].SIndex = feat
	}

	return &xgbin.TreeModel{
		Param: xgbin.TreeParam{
			NumRoots:       1,
			NumNodes:       int32(numNodes),
			NumFeature:     int32(numFeature),
			SizeLeafVector: int32(sizeLeafVector),
		},
		Nodes:      nodes,
		Stats:      stats,
		OutputDim:  sizeLeafVector,
		LeafWeights: parseBaseWeights(obj),
	}, parseXGBTreeCatMeta(obj), nil
}

func parseBaseWeights(obj map[string]json.RawMessage) []float64 {
	raw, ok := obj["base_weights"]
	if !ok {
		return nil
	}
	weights, err := parseFloatArray(raw)
	if err != nil {
		return nil
	}
	return weights
}

type xgbTreeCatMeta struct {
	splitType          []int
	categories         []int32
	categoriesNodes    []int32
	categoriesSegments []int64
	categoriesSizes    []int64
}

func parseXGBTreeCatMeta(obj map[string]json.RawMessage) *xgbTreeCatMeta {
	meta := &xgbTreeCatMeta{}
	if raw, ok := obj["split_type"]; ok {
		if arr, err := parseIntArrayLoose(raw); err == nil {
			meta.splitType = arr
		}
	}
	if raw, ok := obj["categories"]; ok {
		if arr, err := parseInt32Array(raw); err == nil {
			meta.categories = arr
		}
	}
	if raw, ok := obj["categories_nodes"]; ok {
		if arr, err := parseInt32Array(raw); err == nil {
			meta.categoriesNodes = arr
		}
	}
	if raw, ok := obj["categories_segments"]; ok {
		if arr, err := parseInt64Array(raw); err == nil {
			meta.categoriesSegments = arr
		}
	}
	if raw, ok := obj["categories_sizes"]; ok {
		if arr, err := parseInt64Array(raw); err == nil {
			meta.categoriesSizes = arr
		}
	}
	if len(meta.splitType) == 0 {
		return nil
	}
	return meta
}

func parseIntArrayLoose(raw json.RawMessage) ([]int, error) {
	if len(raw) == 0 {
		return nil, fmt.Errorf("empty")
	}
	var ints []int
	if err := json.Unmarshal(raw, &ints); err == nil {
		return ints, nil
	}
	var int32s []int32
	if err := json.Unmarshal(raw, &int32s); err == nil {
		out := make([]int, len(int32s))
		for i, v := range int32s {
			out[i] = int(v)
		}
		return out, nil
	}
	return nil, fmt.Errorf("cannot parse int array")
}

func parseInt64Array(raw json.RawMessage) ([]int64, error) {
	if len(raw) == 0 {
		return nil, fmt.Errorf("empty")
	}
	var out []int64
	if err := json.Unmarshal(raw, &out); err == nil {
		return out, nil
	}
	var ints []int
	if err := json.Unmarshal(raw, &ints); err == nil {
		out = make([]int64, len(ints))
		for i, v := range ints {
			out[i] = int64(v)
		}
		return out, nil
	}
	return nil, fmt.Errorf("cannot parse int64 array")
}

// ---- JSON 字段解析辅助 ----

func parseStringField(raw json.RawMessage) (string, error) {
	if len(raw) == 0 {
		return "", nil
	}
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		return s, nil
	}
	return "", fmt.Errorf("expected string")
}

func parseStringArrayField(raw json.RawMessage) ([]string, error) {
	if len(raw) == 0 {
		return nil, nil
	}
	var arr []string
	if err := json.Unmarshal(raw, &arr); err == nil {
		return arr, nil
	}
	return nil, fmt.Errorf("expected string array")
}

func parseIntField(raw json.RawMessage) (int, error) {
	if len(raw) == 0 {
		return 0, fmt.Errorf("empty int field")
	}
	var i int
	if err := json.Unmarshal(raw, &i); err == nil {
		return i, nil
	}
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		return strconv.Atoi(s)
	}
	var f float64
	if err := json.Unmarshal(raw, &f); err == nil {
		return int(f), nil
	}
	return 0, fmt.Errorf("cannot parse int from %s", string(raw))
}

func parseBaseScore(raw json.RawMessage) (float64, []float64, error) {
	if len(raw) == 0 {
		return 0, nil, nil
	}
	var f float64
	if err := json.Unmarshal(raw, &f); err == nil {
		return f, nil, nil
	}
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		s = strings.TrimSpace(s)
		if strings.HasPrefix(s, "[") && strings.HasSuffix(s, "]") {
			inner := strings.TrimSpace(s[1 : len(s)-1])
			parts := strings.Split(inner, ",")
			scores := make([]float64, 0, len(parts))
			for _, p := range parts {
				p = strings.TrimSpace(p)
				v, err := strconv.ParseFloat(p, 64)
				if err != nil {
					return 0, nil, err
				}
				scores = append(scores, v)
			}
			if len(scores) == 1 {
				return scores[0], scores, nil
			}
			return scores[0], scores, nil
		}
		v, err := strconv.ParseFloat(s, 64)
		return v, nil, err
	}
	return 0, nil, fmt.Errorf("cannot parse base_score")
}

func parseIntArray(raw json.RawMessage) ([]int, error) {
	if len(raw) == 0 {
		return nil, nil
	}
	var arr []int
	if err := json.Unmarshal(raw, &arr); err == nil {
		return arr, nil
	}
	var farr []float64
	if err := json.Unmarshal(raw, &farr); err == nil {
		out := make([]int, len(farr))
		for i, v := range farr {
			out[i] = int(v)
		}
		return out, nil
	}
	return nil, fmt.Errorf("cannot parse int array")
}

func parseInt32Array(raw json.RawMessage) ([]int32, error) {
	arr, err := parseIntArray(raw)
	if err != nil {
		return nil, err
	}
	out := make([]int32, len(arr))
	for i, v := range arr {
		out[i] = int32(v)
	}
	return out, nil
}

func parseFloatArray(raw json.RawMessage) ([]float64, error) {
	if len(raw) == 0 {
		return nil, nil
	}
	var arr []float64
	if err := json.Unmarshal(raw, &arr); err == nil {
		return arr, nil
	}
	var iarr []int
	if err := json.Unmarshal(raw, &iarr); err == nil {
		out := make([]float64, len(iarr))
		for i, v := range iarr {
			out[i] = float64(v)
		}
		return out, nil
	}
	return nil, fmt.Errorf("cannot parse float array")
}

func parseBoolArray(raw json.RawMessage) ([]bool, error) {
	if len(raw) == 0 {
		return nil, nil
	}
	var arr []bool
	if err := json.Unmarshal(raw, &arr); err == nil {
		return arr, nil
	}
	var iarr []int
	if err := json.Unmarshal(raw, &iarr); err == nil {
		out := make([]bool, len(iarr))
		for i, v := range iarr {
			out[i] = v != 0
		}
		return out, nil
	}
	var farr []float64
	if err := json.Unmarshal(raw, &farr); err == nil {
		out := make([]bool, len(farr))
		for i, v := range farr {
			out[i] = v != 0
		}
		return out, nil
	}
	return nil, fmt.Errorf("cannot parse bool array")
}

// adjustBaseScoreForObjective 将 JSON 中概率形式的 base_score 转为 margin（logit）。
// XGBoost 3.x 在 logistic 目标下以概率存储 base_score，推理时先转 logit 再累加树输出。
func adjustBaseScoreForObjective(objective string, forest *tree.ForestIR) {
	switch objective {
	case "binary:logistic", "reg:logistic":
		forest.BaseScore = probabilityToLogit(forest.BaseScore)
		for i, s := range forest.BaseScores {
			forest.BaseScores[i] = probabilityToLogit(s)
		}
	case "reg:gamma", "count:poisson", "reg:tweedie":
		forest.BaseScore = responseMeanToLogMargin(forest.BaseScore)
		for i, s := range forest.BaseScores {
			forest.BaseScores[i] = responseMeanToLogMargin(s)
		}
	}
}

func responseMeanToLogMargin(m float64) float64 {
	if m <= 0 {
		return m
	}
	return math.Log(m)
}

func probabilityToLogit(p float64) float64 {
	if p <= 0 || p >= 1 {
		return p
	}
	return math.Log(p / (1 - p))
}

// ObjectiveToTransform 根据目标函数名推断变换（与根包 IO 逻辑一致）。
func ObjectiveToTransform(objective string, load bool) (tree.TransformType, tree.TransformFn) {
	if !load {
		return tree.TransformRaw, tree.ApplyTransformRaw
	}
	switch objective {
	case "binary:logistic", "reg:logistic":
		return tree.TransformLogistic, tree.ApplyTransformLogistic
	case "multi:softprob":
		return tree.TransformSoftmax, tree.ApplyTransformSoftmax
	case "multi:softmax":
		return tree.TransformArgmax, tree.ApplyTransformArgmax
	case "reg:gamma", "count:poisson", "reg:tweedie":
		return tree.TransformExponential, tree.ApplyTransformExponential
	default:
		return tree.TransformRaw, tree.ApplyTransformRaw
	}
}

// NOutputGroupsForTransform 计算变换后输出维度。
func NOutputGroupsForTransform(nRaw int, outputType tree.TransformType) int {
	if outputType == tree.TransformLeafIndex {
		return nRaw
	}
	if outputType == tree.TransformArgmax {
		return 1
	}
	if outputType == tree.TransformSoftmax {
		return nRaw
	}
	if outputType == tree.TransformRaw || outputType == tree.TransformLogistic || outputType == tree.TransformExponential {
		return nRaw
	}
	return nRaw
}
