package metrics_test

import (
	"math"
	"testing"

	"github.com/linkerlin/leaves/metrics"
)

func TestRMSE(t *testing.T) {
	m := metrics.RMSE{}
	got, err := m.Evaluate([]float64{1, 2, 3}, []float64{1, 3, 2})
	if err != nil {
		t.Fatal(err)
	}
	want := math.Sqrt((0 + 1 + 1) / 3.0)
	if math.Abs(got-want) > 1e-9 {
		t.Errorf("got %f want %f", got, want)
	}
}

func TestLogLoss(t *testing.T) {
	m := metrics.LogLoss{}
	got, err := m.Evaluate([]float64{1, 0}, []float64{0.9, 0.1})
	if err != nil {
		t.Fatal(err)
	}
	if got <= 0 {
		t.Fatalf("expected positive logloss, got %f", got)
	}
}

func TestAUCPerfect(t *testing.T) {
	m := metrics.AUC{}
	got, err := m.Evaluate([]float64{1, 1, 0, 0}, []float64{0.9, 0.8, 0.2, 0.1})
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(got-1.0) > 1e-9 {
		t.Errorf("got %f want 1", got)
	}
}

func TestMLogLoss(t *testing.T) {
	m := metrics.MLogLoss{NumClass: 3}
	// 2 samples, 3 classes row-major
	got, err := m.Evaluate([]float64{0, 2}, []float64{0.7, 0.2, 0.1, 0.1, 0.2, 0.7})
	if err != nil {
		t.Fatal(err)
	}
	if got <= 0 {
		t.Fatalf("expected positive mlogloss, got %f", got)
	}
}

func TestEvaluatePerGroupRMSE(t *testing.T) {
	m := metrics.RMSE{}
	got, err := m.EvaluatePerGroup(
		[]float64{1, 2, 10, 11},
		[]float64{1, 3, 10, 9},
		[]int{2, 2},
	)
	if err != nil {
		t.Fatal(err)
	}
	// group0 rmse=sqrt(0.5), group1 rmse=sqrt(2) → 平均
	want := (math.Sqrt(0.5) + math.Sqrt(2.0)) / 2.0
	if math.Abs(got-want) > 1e-9 {
		t.Errorf("got %f want 1", got)
	}
}
