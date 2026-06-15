package leaves_test

import (
	"testing"

	"github.com/linkerlin/leaves"
	"github.com/linkerlin/leaves/data"
)

func TestTrainBridgeNewLearner(t *testing.T) {
	vals := []float64{0, 1, 1, 0, 0, 1}
	labels := []float64{0, 1, 1}
	dm, err := data.NewDense(vals, 3, 2, labels, nil)
	if err != nil {
		t.Fatal(err)
	}
	learner, err := leaves.NewLearner(leaves.TrainConfig{
		Objective:  leaves.TrainObjectiveSquaredError,
		NumRound:   2,
		MaxDepth:   2,
		TreeMethod: leaves.TrainTreeMethodExact,
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := learner.Fit(dm); err != nil {
		t.Fatal(err)
	}
	if learner.Model() == nil {
		t.Fatal("nil model")
	}
}
