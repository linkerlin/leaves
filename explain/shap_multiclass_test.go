package explain_test

import (
	"math"
	"path/filepath"
	"testing"
	"time"

	"github.com/dmitryikh/leaves/explain"
	"github.com/dmitryikh/leaves/io"
	leafmodel "github.com/dmitryikh/leaves/model"
	"github.com/dmitryikh/leaves/tree"
)

func TestMulticlassLargeModelSHAPFast(t *testing.T) {
	path := filepath.Join("..", "testdata", "lgmulticlass.model")
	m, err := io.LoadFromFile(path, &io.LoadOptions{LoadTransformation: false})
	if err != nil {
		t.Fatal(err)
	}
	f := m.Forest()
	if f == nil || f.NumOutputGroups <= 1 {
		t.Fatal("expected multiclass forest")
	}

	expl := explain.NewTreeExplainer(f)
	x := make([]float64, m.NFeatures())
	x[0] = 0.5
	x[3] = -0.2

	start := time.Now()
	phi, err := expl.ShapleyValuesMulticlass([][]float64{x})
	if err != nil {
		t.Fatal(err)
	}
	if time.Since(start) > 15*time.Second {
		t.Fatalf("SHAP too slow: %v", time.Since(start))
	}

	bases := expl.ExpectedValues()
	raw := make([]float64, m.NRawOutputGroups())
	_ = m.Engine().Predict(x, 0, raw)
	for k := 0; k < f.NumOutputGroups; k++ {
		sum := bases[k]
		for _, v := range phi[0][k] {
			sum += v
		}
		if math.Abs(sum-raw[k]) > 1e-2 {
			t.Errorf("class %d additivity: sum=%f margin=%f", k, sum, raw[k])
		}
	}
}

func TestXGBoostCategoricalE2E(t *testing.T) {
	modelPath := filepath.Join("..", "testdata", "xgboost_categorical_smoke.json")
	m, err := io.LoadFromFile(modelPath, &io.LoadOptions{LoadTransformation: false})
	if err != nil {
		t.Skip(err)
	}
	f := m.Forest()
	if f == nil || len(f.Trees) == 0 {
		t.Fatal("nil forest")
	}
	hasCat := false
	for i := range f.Trees {
		for j := range f.Trees[i].IsCategorical {
			if f.Trees[i].IsCategorical[j] {
				hasCat = true
				break
			}
		}
	}
	if !hasCat {
		t.Fatal("expected categorical splits in model")
	}

	ir := &leafmodel.ModelIR{Forest: f, NOutputGroups: 1, NRawOutputGroups: 1}
	native, err := leafmodel.NewEnsembleFromIR(ir, nil, tree.TransformRaw, tree.BackendNative)
	if err != nil {
		t.Fatal(err)
	}
	simple, err := leafmodel.NewEnsembleFromIR(ir, nil, tree.TransformRaw, tree.BackendBornCPU)
	if err != nil {
		t.Fatal(err)
	}

	cases := [][]float64{{0.5, 1}, {-1.0, 2}, {0.0, 0}}
	for _, x := range cases {
		outN := make([]float64, 1)
		outS := make([]float64, 1)
		_ = native.Engine().Predict(x, 0, outN)
		_ = simple.Engine().Predict(x, 0, outS)
		if math.Abs(outN[0]-outS[0]) > 1e-4 {
			t.Errorf("x=%v native=%f simplego=%f", x, outN[0], outS[0])
		}
	}
}
