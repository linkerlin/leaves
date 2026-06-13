package tree

import (
	"math"
	"testing"
)

func makeConstTree(val float64) *TreeIR {
	return BuildTreeIR(nil, []float64{val}, nil, nil, 0)
}

func TestForestParallelTreeAveraging(t *testing.T) {
	// 2 iterations × 2 parallel trees; leaf values 1,3 and 2,4 → iter avg 2,3 → sum 5
	forest := &ForestIR{
		NumFeatures:     1,
		NumOutputGroups: 1,
		BaseScore:       0.5,
		NumParallelTree: 2,
		IterationIndptr: []int{0, 2, 4},
		TreeInfo:        []int{0, 0, 0, 0},
		Trees: []TreeIR{
			*makeConstTree(1.0),
			*makeConstTree(3.0),
			*makeConstTree(2.0),
			*makeConstTree(4.0),
		},
	}

	got := ForestMargin(forest, []float64{0.0}, 2)
	want := 0.5 + (1.0 + 3.0) + (2.0 + 4.0)
	if math.Abs(got-want) > 1e-9 {
		t.Fatalf("parallel tree sum: got %f want %f", got, want)
	}

	engine := NewNativeEngine(forest, ApplyTransformRaw, TransformRaw, 1)
	raw := make([]float64, 1)
	engine.predictInner([]float64{0.0}, 2, raw, 0)
	if math.Abs(raw[0]-want) > 1e-9 {
		t.Fatalf("native predictInner: got %f want %f", raw[0], want)
	}
}

func TestForestVectorLeaf(t *testing.T) {
	tir := BuildTreeIR(nil, []float64{0.1, 0.2}, nil, nil, 0)
	tir.OutputDim = 2
	forest := &ForestIR{
		NumFeatures:       1,
		NumOutputGroups:   2,
		IterationIndptr:   []int{0, 1},
		Trees:             []TreeIR{*tir},
	}
	m := ForestMargins(forest, []float64{0.0}, 1)
	if len(m) != 2 {
		t.Fatalf("expected 2 margins, got %d", len(m))
	}
	if math.Abs(m[0]-0.1) > 1e-9 || math.Abs(m[1]-0.2) > 1e-9 {
		t.Fatalf("vector leaf margins: got [%f,%f] want [0.1,0.2]", m[0], m[1])
	}
}

func TestForestAverageOutputLGB(t *testing.T) {
	forest := &ForestIR{
		NumFeatures:     1,
		NumOutputGroups: 1,
		BaseScore:       1.0,
		AverageOutput:   true,
		Trees: []TreeIR{
			*makeConstTree(2.0),
			*makeConstTree(4.0),
		},
	}
	got := ForestMargin(forest, []float64{0.0}, 2)
	// base + (2+4)/2 = 1 + 3 = 4
	want := 4.0
	if math.Abs(got-want) > 1e-9 {
		t.Fatalf("average output: got %f want %f", got, want)
	}
}
