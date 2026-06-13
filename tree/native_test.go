package tree

import (
	"math"
	"testing"
)

// 构造一棵简单的数值树用于测试
//        [0] feature=0, threshold=0.5
//        /            \
//    Leaf(0.1)    [1] feature=1, threshold=1.5
//                  /            \
//              Leaf(0.2)     Leaf(0.3)
// 注：lgTree 使用显式子节点索引存储（Left/Right 直接指向 nodes 数组下标或叶子值表下标）
func makeSimpleTree() *TreeIR {
	nodes := []LgNodeData{
		{Feature: 0, Threshold: 0.5, Flags: flagLeftLeaf, Left: 0, Right: 1},
		{Feature: 1, Threshold: 1.5, Flags: flagLeftLeaf | flagRightLeaf, Left: 1, Right: 2},
	}
	leafValues := []float64{0.1, 0.2, 0.3}
	return BuildTreeIR(nodes, leafValues, nil, nil, 0)
}

// 构造一棵常数树（单叶子）
func makeConstantTree() *TreeIR {
	leafValues := []float64{0.42}
	return BuildTreeIR(nil, leafValues, nil, nil, 0)
}

// 构造分类特征树
func makeCategoricalTree() *TreeIR {
	nodes := []LgNodeData{
		{Feature: 0, Threshold: 5.0, Flags: flagCategorical | flagLeftLeaf | flagCatOneHot, Left: 0, Right: 1},
	}
	leafValues := []float64{1.0, -1.0}
	return BuildTreeIR(nodes, leafValues, nil, nil, 1)
}

func makeNonCatTree() *TreeIR {
	// 构造一棵单节点分类树：f0==5 → left(leaf 0)=1.0, else → right(leaf 1)=-1.0
	nodes := []LgNodeData{
		{Feature: 0, Threshold: 5.0, Flags: flagCategorical | flagLeftLeaf | flagRightLeaf | flagCatOneHot, Left: 0, Right: 1},
	}
	leafVals := []float64{1.0, -1.0}
	return BuildTreeIR(nodes, leafVals, nil, nil, 1)
}

func makeForest() *ForestIR {
	trees := []TreeIR{*makeSimpleTree(), *makeSimpleTree()}
	return &ForestIR{
		NumFeatures:     2,
		NumOutputGroups: 1,
		Trees:           trees,
		BaseScore:       0.5,
		WeightDrop:      []float64{1.0, 1.0},
		Name:            "test.gbtree",
	}
}

func TestSimpleTreePrediction(t *testing.T) {
	tree := makeSimpleTree()

	engine := NewNativeEngine(
		&ForestIR{NumFeatures: 2, NumOutputGroups: 1, Trees: []TreeIR{*tree}},
		ApplyTransformRaw, TransformRaw, 1,
	)

	// f0 < 0.5 → 左叶 = 0.1
	p := engine.predictTree(tree, []float64{0.3, 0.0})
	if math.Abs(p-0.1) > 1e-9 {
		t.Errorf("expected 0.1, got %f", p)
	}

	// f0 >= 0.5, f1 < 1.5 → 中叶 = 0.2
	p = engine.predictTree(tree, []float64{0.6, 1.0})
	if math.Abs(p-0.2) > 1e-9 {
		t.Errorf("expected 0.2, got %f", p)
	}

	// f0 >= 0.5, f1 >= 1.5 → 右叶 = 0.3
	p = engine.predictTree(tree, []float64{0.6, 2.0})
	if math.Abs(p-0.3) > 1e-9 {
		t.Errorf("expected 0.3, got %f", p)
	}
}

func TestConstantTreePrediction(t *testing.T) {
	tree := makeConstantTree()
	engine := NewNativeEngine(
		&ForestIR{NumFeatures: 1, NumOutputGroups: 1, Trees: []TreeIR{*tree}},
		ApplyTransformRaw, TransformRaw, 1,
	)

	p := engine.predictTree(tree, []float64{0.0})
	if math.Abs(p-0.42) > 1e-9 {
		t.Errorf("expected 0.42, got %f", p)
	}
}

func TestCategoricalTreePrediction(t *testing.T) {
	tree := makeNonCatTree()
	engine := NewNativeEngine(
		&ForestIR{NumFeatures: 1, NumOutputGroups: 1, Trees: []TreeIR{*tree}},
		ApplyTransformRaw, TransformRaw, 1,
	)

	// f0 == threshold(5.0) → 左叶 = 1.0
	p := engine.predictTree(tree, []float64{5.0})
	if math.Abs(p-1.0) > 1e-9 {
		t.Errorf("expected 1.0, got %f", p)
	}

	// f0 != 5 → 右叶 = -1.0
	p = engine.predictTree(tree, []float64{3.0})
	if math.Abs(p-(-1.0)) > 1e-9 {
		t.Errorf("expected -1.0, got %f", p)
	}
}

func TestMissingValueDefaultLeft(t *testing.T) {
	nodes := []LgNodeData{
		{Feature: 0, Threshold: 0.5, Flags: flagDefaultLeft | flagMissingNan | flagLeftLeaf | flagRightLeaf, Left: 0, Right: 1},
	}
	leafValues := []float64{1.0, -1.0}
	tree := BuildTreeIR(nodes, leafValues, nil, nil, 0)

	engine := NewNativeEngine(
		&ForestIR{NumFeatures: 1, NumOutputGroups: 1, Trees: []TreeIR{*tree}},
		ApplyTransformRaw, TransformRaw, 1,
	)

	// NaN 特征值 → 默认走左 = 1.0
	p := engine.predictTree(tree, []float64{math.NaN()})
	if math.Abs(p-1.0) > 1e-9 {
		t.Errorf("NaN should go left (1.0), got %f", p)
	}
}

func TestMissingValueNaNBecomesZero(t *testing.T) {
	// 无 missingNan 时 NaN 视为 0（XGBoost/LightGBM 兼容）
	nodes := []LgNodeData{
		{Feature: 0, Threshold: -0.5, Flags: flagLeftLeaf | flagRightLeaf, Left: 0, Right: 1},
	}
	leafValues := []float64{1.0, -1.0}
	tree := BuildTreeIR(nodes, leafValues, nil, nil, 0)
	engine := NewNativeEngine(
		&ForestIR{NumFeatures: 1, NumOutputGroups: 1, Trees: []TreeIR{*tree}},
		ApplyTransformRaw, TransformRaw, 1,
	)
	// NaN → 0，0 <= -0.5 为 false，走右 = -1.0
	p := engine.predictTree(tree, []float64{math.NaN()})
	if math.Abs(p-(-1.0)) > 1e-9 {
		t.Errorf("NaN without missingNan should act as 0 (right=-1.0), got %f", p)
	}
}

func TestForestPrediction(t *testing.T) {
	forest := makeForest()
	engine := NewNativeEngine(forest, ApplyTransformRaw, TransformRaw, 1)

	predictions := make([]float64, 1)
	err := engine.Predict([]float64{0.3, 0.0}, 0, predictions)
	if err != nil {
		t.Fatal(err)
	}
	// BaseScore=0.5 + tree1(0.1) + tree2(0.1) = 0.7
	expected := 0.7
	if math.Abs(predictions[0]-expected) > 1e-9 {
		t.Errorf("expected %f, got %f", expected, predictions[0])
	}
}

func TestTransformLogistic(t *testing.T) {
	raw := []float64{0.0}
	out := []float64{0.0}
	ApplyTransformLogistic(raw, out, 0)
	if math.Abs(out[0]-0.5) > 1e-9 {
		t.Errorf("sigmoid(0) = 0.5, got %f", out[0])
	}

	raw[0] = 1.0
	ApplyTransformLogistic(raw, out, 0)
	if math.Abs(out[0]-0.7310585786300049) > 1e-6 {
		t.Errorf("sigmoid(1) ≈ 0.731, got %f", out[0])
	}
}

func TestTransformSoftmax(t *testing.T) {
	raw := []float64{1.0, 2.0, 3.0}
	out := make([]float64, 3)
	ApplyTransformSoftmax(raw, out, 0)

	sum := 0.0
	for _, v := range out {
		sum += v
	}
	if math.Abs(sum-1.0) > 1e-9 {
		t.Errorf("softmax should sum to 1.0, got %f", sum)
	}

	// softmax 应保持顺序
	if out[0] >= out[1] || out[1] >= out[2] {
		t.Errorf("softmax should preserve order")
	}
}

func TestPredictDense(t *testing.T) {
	forest := makeForest()
	engine := NewNativeEngine(forest, ApplyTransformRaw, TransformRaw, 1)

	// 2 samples × 2 features
	vals := []float64{
		0.3, 0.0, // → tree 0.1 each, total 0.5+0.1+0.1=0.7
		0.6, 2.0, // → tree 0.3 each, total 0.5+0.3+0.3=1.1
	}
	predictions := make([]float64, 2)

	err := engine.PredictDense(vals, 2, 2, predictions, 0)
	if err != nil {
		t.Fatal(err)
	}

	exp0 := 0.7
	exp1 := 1.1
	if math.Abs(predictions[0]-exp0) > 1e-9 {
		t.Errorf("sample 0: expected %f, got %f", exp0, predictions[0])
	}
	if math.Abs(predictions[1]-exp1) > 1e-9 {
		t.Errorf("sample 1: expected %f, got %f", exp1, predictions[1])
	}
}

func TestForestIR(t *testing.T) {
	f := makeForest()
	if f.NEstimators() != 2 {
		t.Errorf("NEstimators: expected 2, got %d", f.NEstimators())
	}
	if f.NumFeatures != 2 {
		t.Errorf("NumFeatures: expected 2, got %d", f.NumFeatures)
	}
	nleaves := f.NLeaves()
	if len(nleaves) != 2 || nleaves[0] != 3 || nleaves[1] != 3 {
		t.Errorf("NLeaves: expected [3,3], got %v", nleaves)
	}
}
