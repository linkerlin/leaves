package linear

import (
	"math"
	"testing"

	"github.com/linkerlin/leaves/tree"
)

func TestLinearEnginePredict(t *testing.T) {
	// 2 features, 1 output: y = 0.5 + 1.0*f0 + 2.0*f1
	lin := &LinearIR{
		NumFeatures:     2,
		NumOutputGroups: 1,
		BaseScore:       0.5,
		Weights: []float64{
			1.0, 2.0, // feature weights
			0.0,       // bias
		},
		Name: "test.gblinear",
	}

	engine := NewNativeEngine(lin, tree.ApplyTransformRaw, tree.TransformRaw, 1)

	p := engine.PredictSingle([]float64{1.0, 2.0}, 0)
	want := 0.5 + 1.0 + 4.0
	if math.Abs(p-want) > 1e-9 {
		t.Errorf("expected %f, got %f", want, p)
	}
}

func TestLinearEngineDense(t *testing.T) {
	lin := &LinearIR{
		NumFeatures:     1,
		NumOutputGroups: 1,
		BaseScore:       0.0,
		Weights:         []float64{2.0, 0.0},
	}
	engine := NewNativeEngine(lin, tree.ApplyTransformLogistic, tree.TransformLogistic, 1)

	vals := []float64{0.0, 1.0}
	preds := make([]float64, 2)
	if err := engine.PredictDense(vals, 2, 1, preds, 0); err != nil {
		t.Fatal(err)
	}
	if math.Abs(preds[1]-0.880797) > 1e-5 {
		t.Errorf("expected ~0.881, got %f", preds[1])
	}
}
