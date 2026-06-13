package treebuilder

import (
	"testing"

	"github.com/dmitryikh/leaves/data"
)

func TestBuildGlobalHistBins(t *testing.T) {
	vals := []float64{
		0, 1,
		2, 3,
		4, 5,
		6, 7,
	}
	labels := []float64{0, 1, 0, 1}
	dm, err := data.NewDense(vals, 4, 2, labels, nil)
	if err != nil {
		t.Fatal(err)
	}
	gb := BuildGlobalHistBins(dm, 8, nil)
	if gb == nil {
		t.Fatal("nil global bins")
	}
	cuts, n, ok := gb.Lookup(0)
	if !ok || n < 2 {
		t.Fatalf("feat0 bins=%d ok=%v cuts=%v", n, ok, cuts)
	}
	_, n1, ok1 := gb.Lookup(1)
	if !ok1 || n1 < 2 {
		t.Fatalf("feat1 bins=%d ok=%v", n1, ok1)
	}
}

func TestBuildHistGlobalBinsProducesTree(t *testing.T) {
	dm, idx, grad, hess := synthHistDataset(120, 6)
	cfg := Config{
		MaxDepth:      4,
		LearningRate:  0.3,
		Lambda:        1.0,
		MaxBin:        32,
		Gamma:         0,
		HistBinPolicy: HistBinGlobal,
		GlobalBins:    BuildGlobalHistBins(dm, 32, nil),
	}
	treeIR := BuildHist(dm, idx, grad, hess, cfg)
	if treeIR == nil {
		t.Fatal("nil tree")
	}
}

func TestBuildHistPerNodePolicyStillWorks(t *testing.T) {
	dm, idx, grad, hess := synthHistDataset(80, 4)
	cfg := Config{
		MaxDepth:      3,
		LearningRate:  0.3,
		Lambda:        1.0,
		MaxBin:        16,
		HistBinPolicy: HistBinPerNode,
	}
	treeIR := BuildHist(dm, idx, grad, hess, cfg)
	if treeIR == nil {
		t.Fatal("nil tree")
	}
}

func TestAccelModeCPUForcesPureScan(t *testing.T) {
	histG := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
	histH := make([]float64, len(histG))
	var sumG, sumH float64
	for i := range histG {
		sumG += histG[i]
		sumH += histH[i]
	}
	lambda := 1.0
	ResetAccelStats()
	cfg := Config{AccelMode: AccelModeCPU, UseGPUHist: true}
	sCPU, gCPU := scanHistGainsCPU(histG, histH, sumG, sumH, lambda)
	s, g := scanHistGains(histG, histH, sumG, sumH, lambda, cfg)
	if sCPU != s || gCPU != g {
		t.Fatalf("cpu mode mismatch cpu=(%d,%v) got=(%d,%v)", sCPU, gCPU, s, g)
	}
	sum := AccelSummary()
	if sum == "" {
		t.Fatal("empty accel summary")
	}
}
