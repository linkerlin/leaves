package train_test

import (
	"testing"

	"github.com/dmitryikh/leaves/data"
	"github.com/dmitryikh/leaves/train"
)

func TestLearnerEval(t *testing.T) {
	vals := []float64{0, 1, 1, 0, 0, 1, 1, 0}
	labels := []float64{0, 1, 1, 0}
	dm, err := data.NewDense(vals, 4, 2, labels, nil)
	if err != nil {
		t.Fatal(err)
	}
	learner, err := train.NewLearner(train.Config{
		Objective:  train.ObjectiveSquaredError,
		EvalMetric: train.EvalRMSE,
		NumRound:   5,
		MaxDepth:   2,
		TreeMethod: train.TreeMethodExact,
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := learner.Fit(dm); err != nil {
		t.Fatal(err)
	}
	score, err := learner.Eval(dm)
	if err != nil {
		t.Fatal(err)
	}
	if score < 0 || score > 1 {
		t.Fatalf("unexpected rmse %f", score)
	}
}
