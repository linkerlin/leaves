package train_test

import (
	"testing"

	"github.com/dmitryikh/leaves/data"
	"github.com/dmitryikh/leaves/metrics"
	"github.com/dmitryikh/leaves/train"
)

func TestFitHistParallelThreads(t *testing.T) {
	vals := make([]float64, 400)
	labels := make([]float64, 100)
	for i := 0; i < 100; i++ {
		vals[i*4] = float64(i)
		vals[i*4+1] = float64(i % 11)
		vals[i*4+2] = float64(i % 7)
		vals[i*4+3] = float64(i % 5)
		labels[i] = float64(i % 3)
	}
	dm, err := data.NewDense(vals, 100, 4, labels, nil)
	if err != nil {
		t.Fatal(err)
	}

	single, _ := train.NewLearner(train.Config{
		Objective:    train.ObjectiveSquaredError,
		NumRound:     8,
		MaxDepth:     4,
		LearningRate: 0.3,
		TreeMethod:   train.TreeMethodHist,
		MaxBin:       32,
		NumThreads:   1,
		Seed:         3,
	})
	if err := single.Fit(dm); err != nil {
		t.Fatal(err)
	}

	parallel, _ := train.NewLearner(train.Config{
		Objective:    train.ObjectiveSquaredError,
		NumRound:     8,
		MaxDepth:     4,
		LearningRate: 0.3,
		TreeMethod:   train.TreeMethodHist,
		MaxBin:       32,
		NumThreads:   4,
		Seed:         3,
	})
	if err := parallel.Fit(dm); err != nil {
		t.Fatal(err)
	}

	p1 := make([]float64, 100)
	p2 := make([]float64, 100)
	_ = single.PredictMargins(dm, p1)
	_ = parallel.PredictMargins(dm, p2)
	rmse := metrics.RMSE{}
	for i := range p1 {
		if p1[i] != p2[i] {
			t.Fatalf("thread mismatch at %d: %f vs %f", i, p1[i], p2[i])
		}
	}
	got, _ := rmse.Evaluate(labels, p1)
	if got > 2.0 {
		t.Errorf("rmse too high: %f", got)
	}
}

func TestFitGPUHistBuilds(t *testing.T) {
	vals := []float64{0, 1, 2, 3, 4, 5}
	labels := []float64{0, 1, 2, 3, 4, 5}
	dm, _ := data.NewDense(vals, 6, 1, labels, nil)
	learner, _ := train.NewLearner(train.Config{
		Objective:  train.ObjectiveSquaredError,
		NumRound:   5,
		MaxDepth:   2,
		TreeMethod: train.TreeMethodGPUHist,
	})
	if err := learner.Fit(dm); err != nil {
		t.Fatal(err)
	}
	preds := make([]float64, 6)
	_ = learner.PredictMargins(dm, preds)
	rmse := metrics.RMSE{}
	got, _ := rmse.Evaluate(labels, preds)
	if got > 1.0 {
		t.Errorf("gpu_hist rmse %f", got)
	}
}
