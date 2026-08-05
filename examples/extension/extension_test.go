package main

import (
	"testing"

	"github.com/linkerlin/leaves/v2/data"
	"github.com/linkerlin/leaves/v2/metrics"
	"github.com/linkerlin/leaves/v2/objective"
	"github.com/linkerlin/leaves/v2/train"
)

func TestCustomRegisterTrain(t *testing.T) {
	// init() 已 Register custom:l1 / max_abs_error
	f, err := objective.ByNameWithClass("custom:l1", 0)
	if err != nil {
		t.Fatalf("objective: %v", err)
	}
	if f.Name() != "custom:l1" {
		t.Fatalf("name %q", f.Name())
	}
	m, err := metrics.Resolve("max_abs_error", metrics.Options{})
	if err != nil {
		t.Fatalf("metric: %v", err)
	}
	if m.HigherIsBetter() {
		t.Fatal("max_abs_error should minimize")
	}

	const n, p = 40, 2
	vals := make([]float64, n*p)
	labels := make([]float64, n)
	for i := 0; i < n; i++ {
		x0 := float64(i%5) * 0.2
		x1 := float64(i%3) * 0.3
		vals[i*p+0] = x0
		vals[i*p+1] = x1
		labels[i] = x0 + x1
	}
	dm, err := data.NewDense(vals, n, p, labels, nil)
	if err != nil {
		t.Fatal(err)
	}
	learner, err := train.NewLearner(train.Config{
		Objective:    "custom:l1",
		EvalMetric:   "max_abs_error",
		NumRound:     12,
		MaxDepth:     3,
		LearningRate: 0.25,
		Seed:         1,
		TreeMethod:   "hist",
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
	if score < 0 || score > 10 {
		t.Fatalf("unexpected max_abs_error=%g", score)
	}
	if learner.BoostRounds() <= 0 {
		t.Fatal("no rounds")
	}
}
