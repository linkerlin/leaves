package objective_test

import (
	"math"
	"testing"

	"github.com/linkerlin/leaves/objective"
)

func TestRankListwiseGradHighRelNegative(t *testing.T) {
	preds := []float64{0.0, 0.1, 0.2}
	labels := []float64{3.0, 1.0, 0.0}
	grad := make([]float64, 3)
	hess := make([]float64, 3)
	obj := objective.NewRankListwise(objective.RankTrainConfig{})
	obj.GradHessGroup(preds, labels, nil, grad, hess)
	if grad[0] >= 0 {
		t.Errorf("high-relevance doc wants higher score: grad[0]=%f", grad[0])
	}
	var sum float64
	for _, g := range grad {
		sum += g
	}
	if math.Abs(sum) > 1e-6 {
		t.Errorf("listwise grad should sum to 0, got %f", sum)
	}
}

func TestRankListwiseLossDecreasesAfterGradStep(t *testing.T) {
	preds := []float64{0.0, 0.0, 0.0}
	labels := []float64{2.0, 1.0, 0.0}
	before := objective.ListwiseLoss(preds, labels)

	grad := make([]float64, 3)
	hess := make([]float64, 3)
	obj := objective.NewRankListwise(objective.RankTrainConfig{})
	obj.GradHessGroup(preds, labels, nil, grad, hess)

	step := 0.5
	afterPreds := make([]float64, 3)
	for i := range preds {
		afterPreds[i] = preds[i] - grad[i]/hess[i]*step
	}
	after := objective.ListwiseLoss(afterPreds, labels)
	if after >= before {
		t.Errorf("loss should decrease: before=%f after=%f", before, after)
	}
}

func TestRankListwiseUniformLabelsEqualPreds(t *testing.T) {
	preds := []float64{0.2, 0.2, 0.2}
	labels := []float64{1.0, 1.0, 1.0}
	grad := make([]float64, 3)
	hess := make([]float64, 3)
	obj := objective.NewRankListwise(objective.RankTrainConfig{})
	obj.GradHessGroup(preds, labels, nil, grad, hess)
	var sumG float64
	for _, g := range grad {
		sumG += math.Abs(g)
	}
	if sumG > 1e-6 {
		t.Errorf("uniform labels + equal preds -> ~zero grad, got %v", grad)
	}
}
