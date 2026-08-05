package train_test

import (
	"testing"

	"github.com/linkerlin/leaves/v2/metrics"
	"github.com/linkerlin/leaves/v2/train"
)

func TestEvalMetricResolveMAPE(t *testing.T) {
	cfg := train.Config{
		Objective:  train.ObjectiveSquaredError,
		EvalMetric: train.EvalMAPE,
		NumRound:   1,
	}
	learner, err := train.NewLearner(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if learner.MetricHistory() != nil {
		t.Fatal("unexpected history before fit")
	}
}

func TestMetricsResolveNDCG(t *testing.T) {
	m, err := metrics.Resolve("ndcg@5", metrics.Options{Groups: []int{2, 2}})
	if err != nil {
		t.Fatal(err)
	}
	if m.Name() != "ndcg" {
		t.Fatalf("name %q", m.Name())
	}
}
