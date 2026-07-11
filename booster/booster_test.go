package booster

import (
	"testing"

	"github.com/linkerlin/leaves/data"
	"github.com/linkerlin/leaves/treebuilder"
)

func TestNewGBTreeDefaults(t *testing.T) {
	b := NewGBTree(4, 0.5, 0, treebuilder.Config{}, "", TrainParams{})
	f := b.Forest()
	if f.NumFeatures != 4 || f.NumOutputGroups != 1 {
		t.Fatalf("forest meta = %+v", f)
	}
	if f.Name != "leaves.gbtree" {
		t.Fatalf("name = %q want leaves.gbtree", f.Name)
	}
	if len(f.IterationIndptr) != 1 || f.IterationIndptr[0] != 0 {
		t.Fatalf("IterationIndptr = %v want [0]", f.IterationIndptr)
	}
}

func TestNewGBTreeRFName(t *testing.T) {
	b := NewGBTree(3, 0, 1, treebuilder.Config{}, "", TrainParams{NumParallelTree: 3})
	if f := b.Forest(); f.Name != "leaves.rf" || !f.AverageOutput || f.NumParallelTree != 3 {
		t.Fatalf("rf forest = %+v", f)
	}
}

func TestNewGBTreeDARTName(t *testing.T) {
	b := NewGBTree(3, 0, 1, treebuilder.Config{}, "", TrainParams{DART: &DARTConfig{RateDrop: 0.1}})
	if f := b.Forest(); f.Name != "leaves.dart" {
		t.Fatalf("dart name = %q want leaves.dart", f.Name)
	}
}

func TestNewGBLinearDefaults(t *testing.T) {
	b := NewGBLinear(4, 0, 1.5, GBLinearConfig{})
	lin := b.Linear()
	if lin.NumOutputGroups != 1 {
		t.Fatalf("groups = %d want 1", lin.NumOutputGroups)
	}
	if lin.NumFeatures != 4 {
		t.Fatalf("features = %d want 4", lin.NumFeatures)
	}
	wantW := 4*1 + 1
	if len(lin.Weights) != wantW {
		t.Fatalf("weights len = %d want %d", len(lin.Weights), wantW)
	}
	if b.cfg.LearningRate != 0.5 {
		t.Fatalf("default lr = %v want 0.5", b.cfg.LearningRate)
	}
	if lin.BaseScore != 1.5 {
		t.Fatalf("base = %v want 1.5", lin.BaseScore)
	}
}

func TestSetLearningRateGBLinear(t *testing.T) {
	b := NewGBLinear(2, 1, 0, GBLinearConfig{LearningRate: 0.3})
	SetLearningRate(b, 0.9)
	if b.cfg.LearningRate != 0.9 {
		t.Fatalf("lr = %v want 0.9", b.cfg.LearningRate)
	}
}

// TestGBLinearBoostPredictMargins: zero gradients → no weight update → margins == BaseScore.
func TestGBLinearBoostPredictMargins(t *testing.T) {
	const nf, n, groups, base = 2, 3, 1, 0.25
	vals := []float64{1, 2, 3, 4, 5, 6}
	dm, err := data.NewDense(vals, n, nf, make([]float64, n), nil)
	if err != nil {
		t.Fatal(err)
	}
	b := NewGBLinear(nf, groups, base, GBLinearConfig{LearningRate: 0.1, Lambda: 1.0})
	grad := make([]float64, n*groups)
	hess := make([]float64, n*groups)
	for i := range hess {
		hess[i] = 1.0
	}
	b.Boost(dm, grad, hess)
	out := make([]float64, n*groups)
	b.PredictMargins(dm, out)
	for i, v := range out {
		if v != base {
			t.Fatalf("out[%d] = %v want %v (zero-grad margins should equal base)", i, v, base)
		}
	}
}
