package tree

import "testing"

func TestTruncateToNEstimators(t *testing.T) {
	// 5 轮，每轮 1 树：Indptr [0,1,2,3,4,5]
	f := &ForestIR{
		NumFeatures:     2,
		NumOutputGroups: 1,
		Trees:           make([]TreeIR, 5),
		WeightDrop:      []float64{1, 1, 1, 1, 1},
		TreeInfo:        []int{0, 0, 0, 0, 0},
		IterationIndptr: []int{0, 1, 2, 3, 4, 5},
	}
	f.TruncateToNEstimators(3)
	if f.NEstimators() != 3 {
		t.Fatalf("NEstimators=%d want 3", f.NEstimators())
	}
	if len(f.Trees) != 3 || len(f.WeightDrop) != 3 || len(f.TreeInfo) != 3 {
		t.Fatalf("len trees/wd/info = %d/%d/%d", len(f.Trees), len(f.WeightDrop), len(f.TreeInfo))
	}
	wantIndptr := []int{0, 1, 2, 3}
	if len(f.IterationIndptr) != 4 {
		t.Fatalf("indptr=%v", f.IterationIndptr)
	}
	for i, v := range wantIndptr {
		if f.IterationIndptr[i] != v {
			t.Fatalf("indptr=%v want %v", f.IterationIndptr, wantIndptr)
		}
	}

	// no-op: n >= current
	f.TruncateToNEstimators(10)
	if f.NEstimators() != 3 {
		t.Fatalf("no-op failed: %d", f.NEstimators())
	}
}

func TestTruncateToNEstimatorsMulticlass(t *testing.T) {
	// 3 轮 × 2 class = 6 树；Indptr [0,2,4,6]
	f := &ForestIR{
		NumFeatures:     2,
		NumOutputGroups: 2,
		Trees:           make([]TreeIR, 6),
		WeightDrop:      []float64{1, 1, 1, 1, 1, 1},
		TreeInfo:        []int{0, 1, 0, 1, 0, 1},
		IterationIndptr: []int{0, 2, 4, 6},
	}
	f.TruncateToNEstimators(2)
	if f.NEstimators() != 2 {
		t.Fatalf("NEstimators=%d want 2", f.NEstimators())
	}
	if len(f.Trees) != 4 {
		t.Fatalf("trees=%d want 4", len(f.Trees))
	}
}
