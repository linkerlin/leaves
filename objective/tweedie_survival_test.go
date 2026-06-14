package objective

import (
	"math"
	"testing"
)

func TestTweedieGradHess(t *testing.T) {
	tw := NewTweedie(1.5)
	g, h := tw.GradHess(0, 1, 1)
	if math.IsNaN(g) || math.IsNaN(h) || h <= 0 {
		t.Fatalf("grad=%v hess=%v", g, h)
	}
}

func TestCoxGradHessBatchGolden(t *testing.T) {
	// 对标 XGBoost TestCoxRegressionGPair
	preds := []float64{0, 0.1, 0.9, 1, 0, 0.1, 0.9, 1}
	labels := []float64{0, -2, -2, 2, 3, 5, -10, 100}
	wantG := []float64{0, 0, 0, -0.799, -0.788, -0.590, 0.910, 1.006}
	wantH := []float64{0, 0, 0, 0.160, 0.186, 0.348, 0.610, 0.639}

	grad := make([]float64, len(preds))
	hess := make([]float64, len(preds))
	if err := (Cox{}).GradHessBatch(preds, labels, nil, grad, hess); err != nil {
		t.Fatal(err)
	}
	for i := range preds {
		if math.Abs(grad[i]-wantG[i]) > 0.02 {
			t.Errorf("grad[%d]=%v want %v", i, grad[i], wantG[i])
		}
		if math.Abs(hess[i]-wantH[i]) > 0.02 {
			t.Errorf("hess[%d]=%v want %v", i, hess[i], wantH[i])
		}
	}
}

func TestTweedieByName(t *testing.T) {
	obj, err := ByName("reg:tweedie")
	if err != nil {
		t.Fatal(err)
	}
	if obj.Name() != "reg:tweedie" {
		t.Fatalf("name=%q", obj.Name())
	}
}

func TestSurvivalCoxByName(t *testing.T) {
	obj, err := ByName("survival:cox")
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := IsSurvival(obj); !ok {
		t.Fatal("expected SurvivalFunc")
	}
}
