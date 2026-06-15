package explain_test

import (
	"math"
	"path/filepath"
	"testing"

	_ "github.com/linkerlin/leaves"
	"github.com/linkerlin/leaves/explain"
	"github.com/linkerlin/leaves/io"
	leafmodel "github.com/linkerlin/leaves/model"
)

func TestTreeSHAPAdditivity(t *testing.T) {
	path := filepath.Join("..", "testdata", "lg_breast_cancer.txt")
	m, err := io.LoadFromFile(path, io.DefaultLoadOptions())
	if err != nil {
		t.Fatal(err)
	}
	f := m.Forest()
	if f == nil {
		t.Fatal("nil forest")
	}

	expl := explain.NewTreeExplainer(f)
	x := make([]float64, m.NFeatures())
	phi, err := expl.ShapleyValues([][]float64{x})
	if err != nil {
		t.Fatal(err)
	}

	margin := marginFromModel(m, x)
	sum := expl.ExpectedValue()
	for _, v := range phi[0] {
		sum += v
	}
	if math.Abs(sum-margin) > 1e-4 {
		t.Errorf("additivity: base+shap=%f margin=%f diff=%e", sum, margin, math.Abs(sum-margin))
	}
}

func TestSaabasNonZero(t *testing.T) {
	path := filepath.Join("..", "testdata", "lg_breast_cancer.txt")
	m, err := io.LoadFromFile(path, io.DefaultLoadOptions())
	if err != nil {
		t.Fatal(err)
	}
	expl := explain.NewTreeExplainer(m.Forest())
	x := make([]float64, m.NFeatures())
	phi, err := expl.ApproximateContributions([][]float64{x})
	if err != nil {
		t.Fatal(err)
	}
	total := 0.0
	for _, v := range phi[0] {
		total += v
	}
	if total == 0 {
		t.Fatal("expected non-zero Saabas contributions")
	}
}

func TestModelExplainAPI(t *testing.T) {
	path := filepath.Join("..", "testdata", "lg_breast_cancer.txt")
	m, err := io.LoadFromFile(path, io.DefaultLoadOptions())
	if err != nil {
		t.Fatal(err)
	}
	exp := m.Explain()
	if exp == nil {
		t.Fatal("nil explainer")
	}
	if exp.ExpectedValue() != 0 {
		// LightGBM breast_cancer base is 0
	}
	imp := exp.Importance(explain.ImportanceWeight, nil)
	if imp == nil || len(imp.Scores) == 0 {
		t.Fatal("empty importance")
	}
}

func TestTreeSHAPXGBoostSmoke(t *testing.T) {
	// 与 XGBoost pred_contribs 语义对齐：base + Σφ ≈ margin（interventional，背景=全零）
	path := filepath.Join("..", "testdata", "xgboost_smoke.json")
	m, err := io.LoadFromFile(path, &io.LoadOptions{LoadTransformation: false})
	if err != nil {
		t.Fatal(err)
	}
	exp := m.Explain()
	x := make([]float64, m.NFeatures())
	x[1] = 0.7
	x[5] = 0.6
	phi, err := exp.TreeSHAP([][]float64{x})
	if err != nil {
		t.Fatal(err)
	}
	margin := marginFromModel(m, x)
	sum := exp.ExpectedValue()
	for _, v := range phi[0] {
		sum += v
	}
	if math.Abs(sum-margin) > 1e-4 {
		t.Errorf("pred_contribs additivity: base+shap=%f margin=%f", sum, margin)
	}
}

func marginFromModel(m *leafmodel.Ensemble, x []float64) float64 {
	eng := m.Engine()
	raw := make([]float64, m.NRawOutputGroups())
	_ = eng.Predict(x, 0, raw)
	return raw[0]
}
