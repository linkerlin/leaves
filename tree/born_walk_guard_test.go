//go:build !js

package tree

import (
	"testing"

	borncpu "github.com/born-ml/born/backend/cpu"
	"github.com/born-ml/born/tensor"
)

func TestWalkTreeBatchCatSmallNoPanic(t *testing.T) {
	nodes := []LgNodeData{
		{Feature: 0, Threshold: 3, Flags: flagCategorical | flagLeftLeaf | flagRightLeaf | flagCatSmall, Left: 0, Right: 1},
	}
	tir := BuildTreeIR(nodes, []float64{1.0, -1.0}, nil, nil, 1)
	b := borncpu.New()
	feats, err := tensor.FromSlice([]float64{1, 2, 3, 4}, tensor.Shape{2, 2}, b)
	if err != nil {
		t.Fatal(err)
	}
	out := walkTreeBatch(b, feats, tir)
	if out == nil || len(out.Data()) != 2 {
		t.Fatalf("got %#v", out)
	}
}

func TestBornEnginePredictDenseNoPanicOnNumeric(t *testing.T) {
	forest := makeForest()
	e, err := NewBornEngine(forest, ApplyTransformRaw, TransformRaw, 1, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer e.Close()
	vals := []float64{0.1, 0.2}
	pred := make([]float64, 1)
	if err := e.PredictDense(vals, 1, 2, pred, 0); err != nil {
		t.Fatal(err)
	}
}
