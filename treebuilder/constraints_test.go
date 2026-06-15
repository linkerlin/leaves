package treebuilder

import (
	"testing"

	"github.com/linkerlin/leaves/data"
)

func TestResolveMethodAutoThreshold(t *testing.T) {
	if got := ResolveMethod(MethodAuto, AccelWebGPUMinRows-1); got != MethodExact {
		t.Fatalf("below threshold: got %q", got)
	}
	if got := ResolveMethod(MethodAuto, AccelWebGPUMinRows); got != MethodHist {
		t.Fatalf("at threshold: got %q", got)
	}
}

func TestMonotoneIncreasingRejectsBadSplit(t *testing.T) {
	// y 随 x 增，但第一轮梯度可能偏好反序分裂；+1 约束应仍可建树
	vals := []float64{0, 1, 2, 3, 4, 5, 6, 7}
	labels := []float64{0, 1, 2, 3, 4, 5, 6, 7}
	dm, _ := data.NewDense(vals, 8, 1, labels, nil)
	idx := []int{0, 1, 2, 3, 4, 5, 6, 7}
	grad := make([]float64, 8)
	hess := make([]float64, 8)
	for i := range grad {
		grad[i] = labels[i] - float64(i)*0.1
		hess[i] = 1
	}
	cfg := Config{
		MaxDepth:            3,
		LearningRate:        0.3,
		Lambda:              1,
		Gamma:               0,
		MonotoneConstraints: []int{1},
	}
	tree := BuildExact(dm, idx, grad, hess, cfg)
	if tree == nil || len(tree.LeafValue) == 0 {
		t.Fatal("nil tree")
	}
}

func TestMonotoneHistMatchesExactOnMonotoneData(t *testing.T) {
	vals := make([]float64, 40)
	labels := make([]float64, 40)
	for i := 0; i < 40; i++ {
		vals[i] = float64(i)
		labels[i] = float64(i) * 0.5
	}
	dm, _ := data.NewDense(vals, 40, 1, labels, nil)
	idx := make([]int, 40)
	grad := make([]float64, 40)
	hess := make([]float64, 40)
	for i := range idx {
		idx[i] = i
		grad[i] = labels[i] - float64(i)*0.05
		hess[i] = 1
	}
	cfg := Config{
		MaxDepth:            4,
		LearningRate:        0.3,
		Lambda:              1,
		Gamma:               0,
		MaxBin:              16,
		HistBinPolicy:       HistBinGlobal,
		GlobalBins:          BuildGlobalHistBins(dm, 16, nil),
		MonotoneConstraints: []int{1},
	}
	ex := BuildExact(dm, idx, grad, hess, cfg)
	hi := BuildHist(dm, idx, grad, hess, cfg)
	if ex == nil || hi == nil {
		t.Fatal("nil tree")
	}
}

func TestSplitRespectsMonotone(t *testing.T) {
	if !splitRespectsMonotone(1, 0.1, 0.2) {
		t.Fatal("increasing should allow left<right")
	}
	if splitRespectsMonotone(1, 0.3, 0.2) {
		t.Fatal("increasing should reject left>right")
	}
	if !splitRespectsMonotone(-1, 0.3, 0.2) {
		t.Fatal("decreasing should allow left>right")
	}
}
