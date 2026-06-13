package objective_test

import (
	"math"
	"testing"

	"github.com/dmitryikh/leaves/objective"
)

type rankDM struct {
	rows   int
	labels []float64
}

func (d rankDM) NumRow() int       { return d.rows }
func (d rankDM) Labels() []float64  { return d.labels }
func (d rankDM) Weights() []float64 { return nil }

func TestRankPairwiseGradNonZero(t *testing.T) {
	preds := []float64{0.1, 0.2, 0.3}
	labels := []float64{3, 1, 0}
	grad := make([]float64, 3)
	hess := make([]float64, 3)
	obj := objective.NewRankPairwise(objective.RankTrainConfig{})
	obj.GradHessGroup(preds, labels, nil, grad, hess)
	var sumG float64
	for _, g := range grad {
		sumG += math.Abs(g)
	}
	if sumG < 1e-9 {
		t.Fatalf("expected non-zero grad, got %v", grad)
	}
	var total float64
	for _, g := range grad {
		total += g
	}
	if math.Abs(total) > 1e-6 {
		t.Errorf("grad not balanced: sum=%f grad=%v", total, grad)
	}
}

func TestRankNDCGGradNonZero(t *testing.T) {
	preds := []float64{0.0, 0.1, 0.2}
	labels := []float64{2, 1, 0}
	grad := make([]float64, 3)
	hess := make([]float64, 3)
	obj := objective.NewRankNDCG(objective.RankTrainConfig{LambdaNorm: true})
	obj.GradHessGroup(preds, labels, nil, grad, hess)
	if grad[0] >= 0 {
		t.Errorf("high-relevance doc should get negative grad (loss convention), got %f", grad[0])
	}
}

func TestGradHessRankingLargeGroup(t *testing.T) {
	n := 100
	labels := make([]float64, n)
	preds := make([]float64, n)
	grad := make([]float64, n)
	hess := make([]float64, n)
	for i := range labels {
		labels[i] = float64(i % 4)
		preds[i] = float64(i) * 0.01
	}
	obj := objective.NewRankPairwise(objective.RankTrainConfig{})
	if err := objective.GradHessRanking(obj, rankDM{rows: n, labels: labels}, []int{n}, preds, grad, hess); err != nil {
		t.Fatal(err)
	}
	var sum float64
	for _, g := range grad {
		sum += math.Abs(g)
	}
	if sum < 1e-9 {
		t.Fatal("expected non-zero grad for large group")
	}
}

func TestGradHessRankingMultiGroup(t *testing.T) {
	labels := []float64{3, 0, 2, 0}
	preds := []float64{0.1, 0.2, 0.3, 0.4}
	grad := make([]float64, 4)
	hess := make([]float64, 4)
	obj := objective.NewRankPairwise(objective.RankTrainConfig{})
	if err := objective.GradHessRanking(obj, rankDM{rows: 4, labels: labels}, []int{2, 2}, preds, grad, hess); err != nil {
		t.Fatal(err)
	}
	if grad[0] == 0 && grad[2] == 0 {
		t.Fatalf("expected grad in both groups, got %v", grad)
	}
}
