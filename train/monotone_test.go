package train_test

import (
	"testing"

	"github.com/dmitryikh/leaves/data"
	"github.com/dmitryikh/leaves/train"
)

func TestFitMonotoneIncreasing(t *testing.T) {
	n := 40
	vals := make([]float64, n)
	labels := make([]float64, n)
	for i := 0; i < n; i++ {
		vals[i] = float64(i)
		labels[i] = float64(i) * 2
	}
	dm, err := data.NewDense(vals, n, 1, labels, nil)
	if err != nil {
		t.Fatal(err)
	}
	learner, err := train.NewLearner(train.Config{
		Objective:           train.ObjectiveSquaredError,
		NumRound:            8,
		MaxDepth:            4,
		LearningRate:        0.3,
		TreeMethod:          train.TreeMethodExact,
		MonotoneConstraints: []int{1},
		Seed:                1,
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := learner.Fit(dm); err != nil {
		t.Fatal(err)
	}
	preds := make([]float64, n)
	if err := learner.PredictMargins(dm, preds); err != nil {
		t.Fatal(err)
	}
	for i := 1; i < n; i++ {
		if preds[i]+1e-6 < preds[i-1] {
			t.Fatalf("monotone violation at x=%d: pred[%d]=%v < pred[%d]=%v", i, i, preds[i], i-1, preds[i-1])
		}
	}
}

func TestFitMonotoneHist(t *testing.T) {
	n := 60
	vals := make([]float64, n)
	labels := make([]float64, n)
	for i := 0; i < n; i++ {
		vals[i] = float64(i) * 0.1
		labels[i] = float64(i)
	}
	dm, _ := data.NewDense(vals, n, 1, labels, nil)
	learner, _ := train.NewLearner(train.Config{
		Objective:           train.ObjectiveSquaredError,
		NumRound:            5,
		MaxDepth:            4,
		TreeMethod:          train.TreeMethodHist,
		HistBinPolicy:       "global",
		MonotoneConstraints: []int{-1},
	})
	if err := learner.Fit(dm); err != nil {
		t.Fatal(err)
	}
}
