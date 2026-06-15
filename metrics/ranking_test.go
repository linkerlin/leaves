package metrics_test

import (
	"math"
	"testing"

	"github.com/linkerlin/leaves/metrics"
)

func TestMAPE(t *testing.T) {
	m := metrics.MAPE{}
	got, err := m.Evaluate([]float64{100, 200}, []float64{110, 180})
	if err != nil {
		t.Fatal(err)
	}
	want := (0.1 + 0.1) / 2
	if math.Abs(got-want) > 1e-9 {
		t.Errorf("got %f want %f", got, want)
	}
}

func TestRMSLE(t *testing.T) {
	m := metrics.RMSLE{}
	got, err := m.Evaluate([]float64{0, 1}, []float64{0, 1})
	if err != nil {
		t.Fatal(err)
	}
	if got != 0 {
		t.Errorf("got %f want 0", got)
	}
}

func TestMError(t *testing.T) {
	m := metrics.MError{NumClass: 3}
	got, err := m.Evaluate(
		[]float64{0, 1, 2},
		[]float64{0.9, 0.1, 0.0, 0.1, 0.8, 0.1, 0.1, 0.1, 0.8},
	)
	if err != nil {
		t.Fatal(err)
	}
	if got != 0 {
		t.Errorf("got %f want 0", got)
	}
}

func TestNDCG(t *testing.T) {
	m := metrics.NDCG{RankingMetric: metrics.RankingMetric{Groups: []int{4}, K: 2}}
	got, err := m.Evaluate(
		[]float64{3, 2, 1, 0},
		[]float64{0.9, 0.8, 0.1, 0.2},
	)
	if err != nil {
		t.Fatal(err)
	}
	if got <= 0 || got > 1 {
		t.Fatalf("unexpected ndcg %f", got)
	}
}

func TestMAP(t *testing.T) {
	m := metrics.MAP{RankingMetric: metrics.RankingMetric{Groups: []int{4}}}
	got, err := m.Evaluate(
		[]float64{1, 0, 1, 0},
		[]float64{0.9, 0.8, 0.7, 0.1},
	)
	if err != nil {
		t.Fatal(err)
	}
	if got <= 0 || got > 1 {
		t.Fatalf("unexpected map %f", got)
	}
}
