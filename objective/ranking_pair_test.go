package objective_test

import (
	"math"
	"testing"

	"github.com/linkerlin/leaves/objective"
)

func TestRankPairTopKVsFull(t *testing.T) {
	preds := []float64{0, 0, 0, 0}
	labels := []float64{3, 2, 1, 0}
	fullG := make([]float64, 4)
	fullH := make([]float64, 4)
	topkG := make([]float64, 4)
	topkH := make([]float64, 4)

	full := objective.NewRankPairwise(objective.RankTrainConfig{PairMethod: objective.RankPairFull})
	topk := objective.NewRankPairwise(objective.RankTrainConfig{
		PairMethod:          objective.RankPairTopK,
		NumPairPerSample:    1,
		LambdaNormalization: true,
	})
	full.GradHessGroup(preds, labels, nil, fullG, fullH)
	topk.GradHessGroup(preds, labels, nil, topkG, topkH)

	var sumFull, sumTop float64
	for i := range fullG {
		sumFull += math.Abs(fullG[i])
		sumTop += math.Abs(topkG[i])
	}
	if sumFull <= 0 || sumTop <= 0 {
		t.Fatal("expected non-zero grads")
	}
	if math.Abs(sumFull-sumTop) < 1e-9 {
		t.Fatal("topk1 should sample fewer pairs than full on 4-doc group")
	}
}

func TestRankPairMethodParse(t *testing.T) {
	if objective.ParseRankPairMethod("topk") != objective.RankPairTopK {
		t.Fatal("topk")
	}
	if objective.ParseRankPairMethod("mean") != objective.RankPairMean {
		t.Fatal("mean")
	}
	if objective.ParseRankPairMethod("") != objective.RankPairTopK {
		t.Fatal("default topk")
	}
	if objective.ParseRankPairMethod("full") != objective.RankPairFull {
		t.Fatal("full")
	}
}
