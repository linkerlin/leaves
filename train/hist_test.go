package train_test

import (
	"math"
	"testing"

	"github.com/dmitryikh/leaves/data"
	"github.com/dmitryikh/leaves/metrics"
	"github.com/dmitryikh/leaves/train"
)

func TestFitHistRegression(t *testing.T) {
	vals := []float64{0, 1, 2, 3, 4, 5}
	labels := []float64{0, 1, 2, 3, 4, 5}
	dm, err := data.NewDense(vals, 6, 1, labels, nil)
	if err != nil {
		t.Fatal(err)
	}

	learner, err := train.NewLearner(train.Config{
		Objective:    "reg:squarederror",
		NumRound:     15,
		MaxDepth:     2,
		LearningRate: 0.4,
		TreeMethod:   "hist",
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := learner.Fit(dm); err != nil {
		t.Fatal(err)
	}

	preds := make([]float64, 6)
	if err := learner.PredictMargins(dm, preds); err != nil {
		t.Fatal(err)
	}
	rmse := metrics.RMSE{}
	got, _ := rmse.Evaluate(labels, preds)
	if got > 0.5 {
		t.Errorf("hist RMSE too high: %f", got)
	}

	hist := learner.MetricHistory()
	if len(hist) != 15 {
		t.Fatalf("metric history len %d, want 15", len(hist))
	}
	last := hist[len(hist)-1]
	if math.Abs(last-got) > 1e-6 {
		t.Errorf("last history %f != final rmse %f", last, got)
	}
}

func TestFitBinaryMetricHistory(t *testing.T) {
	vals := []float64{0.1, 0.9, 0.2, 0.8, 0.15, 0.85}
	labels := []float64{0, 1, 0, 1, 0, 1}
	dm, err := data.NewDense(vals, 6, 1, labels, nil)
	if err != nil {
		t.Fatal(err)
	}

	learner, err := train.NewLearner(train.Config{
		Objective:  "binary:logistic",
		NumRound:   5,
		MaxDepth:   2,
		TreeMethod: "hist",
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := learner.Fit(dm); err != nil {
		t.Fatal(err)
	}
	hist := learner.MetricHistory()
	if len(hist) != 5 {
		t.Fatalf("metric history len %d, want 5", len(hist))
	}
	if hist[len(hist)-1] <= 0.5 {
		t.Errorf("expected AUC > 0.5, got %f", hist[len(hist)-1])
	}
}
