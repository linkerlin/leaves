package train_test

import (
	"testing"

	"github.com/linkerlin/leaves/data"
	"github.com/linkerlin/leaves/train"
)

func TestMaxLeavesLimit(t *testing.T) {
	vals := make([]float64, 40)
	labels := make([]float64, 20)
	for i := 0; i < 20; i++ {
		vals[i*2] = float64(i)
		vals[i*2+1] = float64(i % 5)
		labels[i] = float64(i % 3)
	}
	dm, err := data.NewDense(vals, 20, 2, labels, nil)
	if err != nil {
		t.Fatal(err)
	}
	const maxLeaves = 4
	learner, err := train.NewLearner(train.Config{
		Objective:  train.ObjectiveSquaredError,
		NumRound:   3,
		MaxDepth:   8,
		MaxLeaves:  maxLeaves,
		TreeMethod: train.TreeMethodHist,
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := learner.Fit(dm); err != nil {
		t.Fatal(err)
	}
	for i, tr := range learner.Model().Forest.Trees {
		if tr.NumLeaves > maxLeaves {
			t.Fatalf("tree %d leaves=%d > max %d", i, tr.NumLeaves, maxLeaves)
		}
	}
}
