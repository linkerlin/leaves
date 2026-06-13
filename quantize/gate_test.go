package quantize_test

import (
	"strings"
	"testing"

	"github.com/dmitryikh/leaves/quantize"
	"github.com/dmitryikh/leaves/tree"
)

func TestParityGatePassFail(t *testing.T) {
	pass := quantize.ParityResult{Samples: 10, MaxMarginDiff: 0.01, Failures: 0}
	if !pass.Pass(quantize.Gate{MaxMarginDiff: 0.15, MaxFailRate: 0.02}) {
		t.Fatal("expected pass")
	}
	fail := quantize.ParityResult{Samples: 10, MaxMarginDiff: 0.2, Failures: 1}
	if fail.Pass(quantize.Gate{MaxMarginDiff: 0.15, MaxFailRate: 0.0}) {
		t.Fatal("expected fail on margin")
	}
	if fail.Pass(quantize.Gate{MaxMarginDiff: 0.25, MaxFailRate: 0.0}) {
		t.Fatal("expected fail on fail rate")
	}
	if !fail.Pass(quantize.Gate{MaxMarginDiff: 0.25, MaxFailRate: 0.2}) {
		t.Fatal("expected pass with loose gate")
	}
}

func TestCheckParityWithGateDetectsMismatch(t *testing.T) {
	tr1 := tree.BuildTreeIR(nil, []float64{0.2}, nil, nil, 0)
	tr2 := tree.BuildTreeIR(nil, []float64{0.9}, nil, nil, 0)
	orig := &tree.ForestIR{
		NumFeatures: 1, NumOutputGroups: 1,
		Trees: []tree.TreeIR{*tr1}, BaseScore: 0,
	}
	other := &tree.ForestIR{
		NumFeatures: 1, NumOutputGroups: 1,
		Trees: []tree.TreeIR{*tr2}, BaseScore: 0,
	}
	qf, err := quantize.QuantizeForest(other, quantize.Config{})
	if err != nil {
		t.Fatal(err)
	}
	_, err = quantize.CheckParityWithGate(orig, qf, [][]float64{{0}}, 0, quantize.Gate{
		MaxMarginDiff: 1e-9,
		MaxFailRate:   0,
	})
	if err == nil || !strings.Contains(err.Error(), "parity gate failed") {
		t.Fatalf("expected gate error, got %v", err)
	}
}

func TestCheckParityNilForest(t *testing.T) {
	r := quantize.CheckParity(nil, nil, nil, 0)
	if r.Samples != 0 || r.Pass(quantize.DefaultGate()) {
		t.Fatalf("nil parity: %+v", r)
	}
}
