package treebuilder

import (
	"math"
	"testing"

	"github.com/linkerlin/leaves/v2/data"
)

// TestBuildVectorLeaf 验证向量叶（multi_output_tree）建树的数学正确性。
// 构造 4 样本 × 1 特征 × k=2 的确定性场景，手算最优分裂与叶向量。
//
// 特征值 [0,0,1,1]；grad = [[1,0],[1,0],[-1,0],[-1,0]]，hess 全 [1,1]。
// 在 thr=0.5 分裂：left={0,1} sumG=[2,0]；right={2,3} sumG=[-2,0]。
// 多输出增益（lambda=1）= Σ_c 0.5*(gl²/(hl+λ)+gr²/(hr+λ)-g²/(h+λ))：
//
//	c=0: 0.5*(4/3 + 4/3 - 0) = 4/3；c=1: 0 → 总增益 4/3 > 0 → 分裂。
//
// 叶权重 = -G_c/(H_c+λ)：left=[-2/3, 0]，right=[2/3, 0]。
func TestBuildVectorLeaf(t *testing.T) {
	vals := []float64{0, 0, 1, 1}
	dm, err := data.NewDense(vals, 4, 1, make([]float64, 4), nil)
	if err != nil {
		t.Fatal(err)
	}
	// grad/hess 布局 [n*k] 行主序。
	grad := []float64{1, 0, 1, 0, -1, 0, -1, 0}
	hess := []float64{1, 1, 1, 1, 1, 1, 1, 1}
	cfg := Config{OutputDim: 2, MaxDepth: 1, Lambda: 1, LearningRate: 1, Gamma: 0, MinHessian: 1e-6}

	tir := BuildExact(dm, []int{0, 1, 2, 3}, grad, hess, cfg)
	if tir.OutputDim != 2 {
		t.Fatalf("OutputDim = %d, want 2", tir.OutputDim)
	}
	if tir.NumNodes != 1 {
		t.Fatalf("NumNodes = %d, want 1 (single split)", tir.NumNodes)
	}
	if tir.SplitFeature[0] != 0 {
		t.Fatalf("SplitFeature = %d, want 0", tir.SplitFeature[0])
	}
	if math.Abs(tir.SplitThreshold[0]-0.5) > 1e-9 {
		t.Fatalf("SplitThreshold = %v, want 0.5", tir.SplitThreshold[0])
	}
	wantLeaves := []float64{-2.0 / 3.0, 0, 2.0 / 3.0, 0}
	if len(tir.LeafValue) != len(wantLeaves) {
		t.Fatalf("len(LeafValue) = %d, want %d", len(tir.LeafValue), len(wantLeaves))
	}
	for i, w := range wantLeaves {
		if math.Abs(tir.LeafValue[i]-w) > 1e-9 {
			t.Fatalf("LeafValue[%d] = %v, want %v (full = %v)", i, tir.LeafValue[i], w, tir.LeafValue)
		}
	}
}

// TestBuildScalarMatchesK1 k=1 向量路径与标量语义等价（回归保护）。
func TestBuildScalarMatchesK1(t *testing.T) {
	vals := []float64{0, 0, 1, 1}
	dm, err := data.NewDense(vals, 4, 1, make([]float64, 4), nil)
	if err != nil {
		t.Fatal(err)
	}
	grad := []float64{1, 1, -1, -1}
	hess := []float64{1, 1, 1, 1}
	cfg := Config{MaxDepth: 1, Lambda: 1, LearningRate: 1, Gamma: 0, MinHessian: 1e-6}

	tir := BuildExact(dm, []int{0, 1, 2, 3}, grad, hess, cfg)
	if tir.OutputDim != 1 {
		t.Fatalf("OutputDim = %d, want 1", tir.OutputDim)
	}
	// 标量叶：left=-2/3，right=2/3（与 k=2 的 c=0 分量一致）。
	want := []float64{-2.0 / 3.0, 2.0 / 3.0}
	if len(tir.LeafValue) != 2 {
		t.Fatalf("len(LeafValue) = %d, want 2", len(tir.LeafValue))
	}
	for i, w := range want {
		if math.Abs(tir.LeafValue[i]-w) > 1e-9 {
			t.Fatalf("LeafValue[%d] = %v, want %v", i, tir.LeafValue[i], w)
		}
	}
}
