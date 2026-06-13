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

func TestGPUHistFallbackWithoutTag(t *testing.T) {
	if BornHistAvailable() {
		t.Skip("born_train tag enabled")
	}
	dm, idx, grad, hess := synthHistDataset(50, 4)
	cfg := Config{MaxDepth: 3, MaxBin: 16}
	if BuildHistGPU(dm, idx, grad, hess, cfg) != nil {
		t.Fatal("expected nil without born_train")
	}
	tree := Build(dm, idx, grad, hess, cfg, MethodGPUHist)
	if tree == nil {
		t.Fatal("gpu_hist should fall back to CPU hist")
	}
}

func TestResolveMethodGPUHist(t *testing.T) {
	if ResolveMethod(MethodGPUHist, 1000) != MethodGPUHist {
		t.Fatal("gpu_hist not preserved")
	}
}
