package model

import (
	"strings"
	"testing"

	"github.com/linkerlin/leaves/tree"
)

func constantForest(base float64, leaf float64) *tree.ForestIR {
	tr := tree.BuildTreeIR(nil, []float64{leaf}, nil, nil, 0)
	return &tree.ForestIR{
		NumFeatures:     1,
		NumOutputGroups: 1,
		Trees:           []tree.TreeIR{*tr},
		BaseScore:       base,
	}
}

func TestReloadWithoutRegisteredLoader(t *testing.T) {
	old := registeredReloadLoader
	t.Cleanup(func() { registeredReloadLoader = old })
	registeredReloadLoader = nil

	f := constantForest(0.5, 0.1)
	eng := tree.NewNativeEngine(f, tree.ApplyTransformRaw, tree.TransformRaw, 1)
	e := NewEnsemble(eng)

	err := e.Reload("any.json", nil)
	if err == nil || !strings.Contains(err.Error(), "not registered") {
		t.Fatalf("expected not registered error, got %v", err)
	}
}

func TestReplaceEngineSwapsPredictions(t *testing.T) {
	eng1 := tree.NewNativeEngine(constantForest(0.5, 0.1), tree.ApplyTransformRaw, tree.TransformRaw, 1)
	eng2 := tree.NewNativeEngine(constantForest(1.0, 0.2), tree.ApplyTransformRaw, tree.TransformRaw, 1)
	e := NewEnsemble(eng1)

	before := e.PredictSingle([]float64{0.0}, 0)
	if err := e.ReplaceEngine(eng2); err != nil {
		t.Fatal(err)
	}
	after := e.PredictSingle([]float64{0.0}, 0)
	if before == after {
		t.Fatalf("predictions should differ after ReplaceEngine: %v vs %v", before, after)
	}
}

func TestDetachEnginePreventsDoubleClose(t *testing.T) {
	eng := tree.NewNativeEngine(constantForest(0.0, 0.3), tree.ApplyTransformRaw, tree.TransformRaw, 1)
	e := NewEnsemble(eng)
	detached := e.DetachEngine()
	if e.Engine() != nil {
		t.Fatal("ensemble engine should be nil after detach")
	}
	if detached == nil {
		t.Fatal("detached engine nil")
	}
	if err := e.Close(); err != nil {
		t.Fatal(err)
	}
	if err := detached.Close(); err != nil {
		t.Fatal(err)
	}
}
