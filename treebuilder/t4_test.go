package treebuilder

import (
	"testing"

	"github.com/dmitryikh/leaves/data"
)

func synthHistDataset(rows, cols int) (data.Matrix, []int, []float64, []float64) {
	vals := make([]float64, rows*cols)
	labels := make([]float64, rows)
	for i := 0; i < rows; i++ {
		labels[i] = float64(i % 7)
		for j := 0; j < cols; j++ {
			vals[i*cols+j] = float64(i)*0.01 + float64(j)*0.13 + float64((i+j)%5)
		}
	}
	dm, _ := data.NewDense(vals, rows, cols, labels, nil)
	idx := make([]int, rows)
	for i := range idx {
		idx[i] = i
	}
	grad := make([]float64, rows)
	hess := make([]float64, rows)
	for i := range grad {
		grad[i] = labels[i] - float64(i%3)
		hess[i] = 1
	}
	return dm, idx, grad, hess
}

func TestParallelHistMatchesSingleThread(t *testing.T) {
	dm, idx, grad, hess := synthHistDataset(200, 8)
	cfg := Config{MaxDepth: 4, LearningRate: 0.3, Lambda: 1.0, MaxBin: 32, Gamma: 0}

	single := BuildHist(dm, idx, grad, hess, cfg)
	cfg.NumThreads = 4
	parallel := BuildHist(dm, idx, grad, hess, cfg)
	if single == nil || parallel == nil {
		t.Fatal("nil tree")
	}
	if single.NumLeaves != parallel.NumLeaves {
		t.Errorf("leaves single=%d parallel=%d", single.NumLeaves, parallel.NumLeaves)
	}
	if len(single.LeafValue) > 0 && len(parallel.LeafValue) > 0 {
		for i := range single.LeafValue {
			if i >= len(parallel.LeafValue) {
				break
			}
			if single.LeafValue[i] != parallel.LeafValue[i] {
				t.Errorf("leaf[%d] single=%f parallel=%f", i, single.LeafValue[i], parallel.LeafValue[i])
				break
			}
		}
	}
}

func TestScanHistGainsCPU(t *testing.T) {
	histG := []float64{1, 2, -1, 0.5}
	histH := []float64{1, 1, 1, 1}
	s, g := scanHistGainsCPU(histG, histH, 2.5, 4, 1)
	if s < 0 || g <= 0 {
		t.Errorf("expected positive gain, got split=%d gain=%f", s, g)
	}
}

func TestScanHistGainsBornMatchesCPU(t *testing.T) {
	if !BornHistAvailable() {
		t.Skip("born hist unavailable")
	}
	histG := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
	histH := make([]float64, len(histG))
	for i := range histH {
		histH[i] = 1
	}
	var sumG, sumH float64
	for i := range histG {
		sumG += histG[i]
		sumH += histH[i]
	}
	lambda := 1.0
	sCPU, gCPU := scanHistGainsCPU(histG, histH, sumG, sumH, lambda)
	sBorn, gBorn := scanHistGains(histG, histH, sumG, sumH, lambda, Config{})
	if sCPU != sBorn || gCPU != gBorn {
		t.Errorf("cpu=(%d,%v) born=(%d,%v)", sCPU, gCPU, sBorn, gBorn)
	}
}

func TestGPUHistBuildsTree(t *testing.T) {
	dm, idx, grad, hess := synthHistDataset(50, 4)
	cfg := Config{MaxDepth: 3, MaxBin: 16}
	treeIR := BuildHistGPU(dm, idx, grad, hess, cfg)
	if treeIR == nil {
		t.Fatal("gpu_hist should build tree with accel fallback chain")
	}
	built := Build(dm, idx, grad, hess, cfg, MethodGPUHist)
	if built == nil {
		t.Fatal("gpu_hist via Build should not be nil")
	}
}

func TestResolveMethodGPUHist(t *testing.T) {
	if ResolveMethod(MethodGPUHist, 1000) != MethodGPUHist {
		t.Fatal("gpu_hist not preserved")
	}
}
