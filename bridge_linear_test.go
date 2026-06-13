package leaves

import (
	"math"
	"path/filepath"
	"testing"
)

// TestBridgeXGBLinear 验证 gblinear 经 LinearIR 桥接后预测正确。
func TestBridgeXGBLinear(t *testing.T) {
	modelPath := filepath.Join("testdata", "xgblin_agaricus.model")

	model, err := XGBLinearFromFile(modelPath, true)
	if err != nil {
		t.Fatalf("load gblinear: %v", err)
	}

	engine, err := NewEngineFromEnsemble(model)
	if err != nil {
		t.Fatalf("new engine: %v", err)
	}
	defer engine.Close()

	nFeatures := engine.NFeatures()
	fvals := make([]float64, nFeatures)

	oldPred := model.PredictSingle(fvals, 0)
	newPred := engine.PredictSingle(fvals, 0)

	diff := math.Abs(oldPred - newPred)
	if diff > 1e-6 {
		t.Errorf("gblinear predict mismatch: old=%f new=%f (diff=%e)", oldPred, newPred, diff)
	}

	rows := 50
	vals := make([]float64, rows*nFeatures)
	oldPreds := make([]float64, rows*model.NOutputGroups())
	newPreds := make([]float64, rows*engine.NOutputGroups())

	_ = model.PredictDense(vals, rows, nFeatures, oldPreds, 0, 0)
	_ = engine.PredictDense(vals, rows, nFeatures, newPreds, 0)

	for i := 0; i < len(oldPreds); i++ {
		diff := math.Abs(oldPreds[i] - newPreds[i])
		if diff > 1e-6 {
			t.Errorf("batch mismatch at %d: old=%f new=%f (diff=%e)", i, oldPreds[i], newPreds[i], diff)
		}
	}
}

// TestEnsembleToModelIR 验证 ModelIR 分化。
func TestEnsembleToModelIR(t *testing.T) {
	linPath := filepath.Join("testdata", "xgblin_agaricus.model")
	linModel, err := XGBLinearFromFile(linPath, false)
	if err != nil {
		t.Fatal(err)
	}
	ir := EnsembleToModelIR(linModel)
	if ir == nil || ir.Linear == nil || ir.Forest != nil {
		t.Fatalf("expected Linear ModelIR, got %+v", ir)
	}
	if len(ir.Linear.Weights) == 0 {
		t.Fatal("expected non-empty weights")
	}

	treePath := filepath.Join("testdata", "xgagaricus.model")
	treeModel, err := XGEnsembleFromFile(treePath, false)
	if err != nil {
		t.Fatal(err)
	}
	ir2 := EnsembleToModelIR(treeModel)
	if ir2 == nil || ir2.Forest == nil {
		t.Fatal("expected Forest ModelIR")
	}
}
