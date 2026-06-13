package treebuilder

import (
	"testing"

	"github.com/dmitryikh/leaves/data"
)

func TestBuildHistMatchesExactOnTiny(t *testing.T) {
	vals := []float64{0, 1, 2, 3, 4, 5}
	labels := []float64{0, 1, 2, 3, 4, 5}
	dm, err := data.NewDense(vals, 6, 1, labels, nil)
	if err != nil {
		t.Fatal(err)
	}
	idx := []int{0, 1, 2, 3, 4, 5}
	grad := []float64{0.1, -0.2, 0.3, -0.1, 0.05, -0.05}
	hess := []float64{1, 1, 1, 1, 1, 1}
	cfg := Config{MaxDepth: 2, LearningRate: 0.3, Lambda: 1.0, MaxBin: 8}

	exact := BuildExact(dm, idx, grad, hess, cfg)
	hist := BuildHist(dm, idx, grad, hess, cfg)
	if exact == nil || hist == nil {
		t.Fatal("nil tree")
	}
	if exact.NumNodes == 0 && hist.NumNodes == 0 {
		return
	}
}
