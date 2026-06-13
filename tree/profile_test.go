package tree

import (
	"math"
	"testing"
)

func TestProfileWalkStats(t *testing.T) {
	f := makeForest()
	stats := ProfileWalkStats(f, []float64{0.3, 0.0}, 0)
	if stats.Trees != 2 {
		t.Fatalf("trees: %d", stats.Trees)
	}
	// 两棵树均 1 步到左叶
	if stats.Steps != 2 {
		t.Fatalf("steps: %d want 2", stats.Steps)
	}
	stats2 := ProfileWalkStats(f, []float64{0.6, 1.0}, 0)
	if stats2.Steps != 4 {
		t.Fatalf("steps: %d want 4", stats2.Steps)
	}
}

func TestProfileNativeDense(t *testing.T) {
	f := makeForest()
	engine := NewNativeEngine(f, ApplyTransformRaw, TransformRaw, 1)

	vals := []float64{
		0.3, 0.0,
		0.6, 1.0,
	}
	preds := make([]float64, 2)
	prof, err := ProfileNativeDense(engine, vals, 2, 2, preds, 0)
	if err != nil {
		t.Fatal(err)
	}
	if prof.Rows != 2 {
		t.Fatalf("rows: %d", prof.Rows)
	}
	if prof.Elapsed < 0 {
		t.Fatal("negative elapsed")
	}
	if prof.TreesPerSample != 2 {
		t.Fatalf("trees per sample: %d", prof.TreesPerSample)
	}
	if prof.TotalWalkSteps != 6 {
		t.Fatalf("walk steps: %d want 6", prof.TotalWalkSteps)
	}
	if math.Abs(prof.AvgStepsPerTree-1.5) > 1e-9 {
		t.Fatalf("avg steps: %v want 1.5", prof.AvgStepsPerTree)
	}
}
