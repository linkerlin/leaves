package objective

import (
	"math"
	"testing"

	"github.com/linkerlin/leaves/data"
)

func TestAFTIntervalGradFinite(t *testing.T) {
	iv := data.AFTInterval{Lower: 1, Upper: 5}
	const eps = 1e-6
	p := 0.3
	g, _ := aftIntervalGradHess(p, 1, iv)
	f := func(x float64) float64 {
		mu := math.Exp(x)
		aL := math.Exp(-iv.Lower / mu)
		aU := math.Exp(-iv.Upper / mu)
		diff := aL - aU
		if diff < 1e-16 {
			diff = 1e-16
		}
		return -math.Log(diff)
	}
	num := (f(p+eps) - f(p-eps)) / (2 * eps)
	if math.Abs(g-num) > 1e-3 {
		t.Fatalf("interval grad analytic=%f numeric=%f", g, num)
	}
}

func TestAFTGradHessBatchMatchesScalar(t *testing.T) {
	aft := AFTNormal{}
	labels := []float64{2, -3}
	preds := []float64{0.1, -0.2}
	grad := make([]float64, 2)
	hess := make([]float64, 2)
	if err := aft.GradHessBatch(preds, labels, nil, grad, hess); err != nil {
		t.Fatal(err)
	}
	for i, y := range labels {
		g, h := aft.GradHess(preds[i], y, 1)
		if math.Abs(g-grad[i]) > 1e-12 || math.Abs(h-hess[i]) > 1e-12 {
			t.Fatalf("i=%d scalar (%f,%f) batch (%f,%f)", i, g, h, grad[i], hess[i])
		}
	}
}
