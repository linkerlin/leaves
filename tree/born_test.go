package tree

import (
	"math"
	"testing"
)

func TestBornEngineSmoke(t *testing.T) {
	forest := makeForest()
	engine, err := NewBornEngine(forest, ApplyTransformRaw, TransformRaw, 1, nil)
	if err != nil {
		t.Fatalf("NewBornEngine: %v", err)
	}
	defer engine.Close()

	fvals := []float64{0.5, 1.5}
	got := engine.PredictSingle(fvals, 0)
	native := NewNativeEngine(forest, ApplyTransformRaw, TransformRaw, 1)
	want := native.PredictSingle(fvals, 0)
	if math.Abs(got-want) > 1e-9 {
		t.Errorf("born vs native: got %v want %v", got, want)
	}
}

func TestBornEngineDenseParity(t *testing.T) {
	forest := makeForest()
	born, err := NewBornEngine(forest, ApplyTransformRaw, TransformRaw, 1, nil)
	if err != nil {
		t.Fatalf("NewBornEngine: %v", err)
	}
	defer born.Close()
	native := NewNativeEngine(forest, ApplyTransformRaw, TransformRaw, 1)

	vals := []float64{0.5, 1.5, 2.5, 3.5}
	predsB := make([]float64, 2)
	predsN := make([]float64, 2)
	_ = born.PredictDense(vals, 2, 2, predsB, 0)
	_ = native.PredictDense(vals, 2, 2, predsN, 0)
	for i := range predsB {
		if math.Abs(predsB[i]-predsN[i]) > 1e-9 {
			t.Errorf("row %d: born=%v native=%v", i, predsB[i], predsN[i])
		}
	}
}

func TestBornWebGPUParitySmoke(t *testing.T) {
	if !BornWebGPUAvailable() {
		t.Skip("webgpu not available")
	}
	forest := makeForest()
	gpu, err := NewBornEngine(forest, ApplyTransformRaw, TransformRaw, 1, &BornConfig{UseGPU: true})
	if err != nil {
		t.Fatal(err)
	}
	defer gpu.Close()
	if !gpu.BornUsingGPU() {
		t.Skip("webgpu init failed")
	}
	cpu, err := NewBornEngine(forest, ApplyTransformRaw, TransformRaw, 1, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer cpu.Close()
	native := NewNativeEngine(forest, ApplyTransformRaw, TransformRaw, 1)

	vals := []float64{0.5, 1.5, 2.5, 3.5}
	for _, eng := range []*BornEngine{cpu, gpu} {
		label := "cpu"
		if eng.BornUsingGPU() {
			label = "gpu"
		}
		got := make([]float64, 2)
		want := make([]float64, 2)
		_ = eng.PredictDense(vals, 2, 2, got, 0)
		_ = native.PredictDense(vals, 2, 2, want, 0)
		for i := range got {
			if math.Abs(got[i]-want[i]) > 1e-5 {
				t.Errorf("%s row %d: got %v want %v", label, i, got[i], want[i])
			}
		}
	}
}

func TestBornWebGPUAvailable(t *testing.T) {
	t.Logf("BornWebGPUAvailable=%v", BornWebGPUAvailable())
}
