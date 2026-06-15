package objective_test

import (
	"math"
	"testing"

	"github.com/linkerlin/leaves/objective"
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

func TestRankPairwiseGradReference(t *testing.T) {
	// 单对 (rel 高 vs 低)，pred 相等 → σ(0)=0.5，λ=(1-σ)=0.5
	preds := []float64{0, 0}
	labels := []float64{2, 0}
	grad := make([]float64, 2)
	hess := make([]float64, 2)
	obj := objective.NewRankPairwise(objective.RankTrainConfig{})
	obj.GradHessGroup(preds, labels, nil, grad, hess)
	wantG := []float64{-0.5, 0.5}
	wantH := []float64{0.25, 0.25}
	for i := range grad {
		if math.Abs(grad[i]-wantG[i]) > 1e-12 {
			t.Errorf("grad[%d]: got %f want %f", i, grad[i], wantG[i])
		}
		if math.Abs(hess[i]-wantH[i]) > 1e-12 {
			t.Errorf("hess[%d]: got %f want %f", i, hess[i], wantH[i])
		}
	}
}

func TestRankPairwiseSkipsTiesAndLowerRel(t *testing.T) {
	preds := []float64{0.1, 0.2, 0.3}
	labels := []float64{1, 1, 1} // 全 tie：无有效 pair
	grad := make([]float64, 3)
	hess := make([]float64, 3)
	obj := objective.NewRankPairwise(objective.RankTrainConfig{})
	obj.GradHessGroup(preds, labels, nil, grad, hess)
	for i, g := range grad {
		if g != 0 {
			t.Errorf("tie group grad[%d]=%f want 0", i, g)
		}
	}
}

func TestRankPairwiseNoNDCGScale(t *testing.T) {
	preds := []float64{0.0, 0.5}
	labels := []float64{3, 0}
	gPW := make([]float64, 2)
	hPW := make([]float64, 2)
	gND := make([]float64, 2)
	hND := make([]float64, 2)
	pw := objective.NewRankPairwise(objective.RankTrainConfig{})
	nd := objective.NewRankNDCG(objective.RankTrainConfig{LambdaNorm: true, NDCGK: 2})
	pw.GradHessGroup(preds, labels, nil, gPW, hPW)
	nd.GradHessGroup(preds, labels, nil, gND, hND)
	if math.Abs(gPW[0]-gND[0]) < 1e-9 && math.Abs(gPW[1]-gND[1]) < 1e-9 {
		t.Fatal("pairwise and ndcg grads should differ when ΔNDCG scale applies")
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
