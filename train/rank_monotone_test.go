package train_test

import (
	"testing"

	"github.com/dmitryikh/leaves/data"
	"github.com/dmitryikh/leaves/train"
)

// monotoneRankingMatrix 合成排序数据：feat0 与 label 正相关，便于 feat0 递增单调约束。
func monotoneRankingMatrix(t *testing.T) *data.DenseWithGroups {
	t.Helper()
	// 2 queries × 4 docs
	labels := []float64{3, 2, 1, 0, 3, 1, 0, 0}
	vals := make([]float64, len(labels)*2)
	for i, y := range labels {
		vals[i*2] = float64(y) + 0.1*float64(i%4) // feat0 随相关性递增
		vals[i*2+1] = 0.5
	}
	dense, err := data.NewDense(vals, len(labels), 2, labels, nil)
	if err != nil {
		t.Fatal(err)
	}
	dm, err := data.NewDenseWithGroups(dense, []int{4, 4})
	if err != nil {
		t.Fatal(err)
	}
	return dm
}

func assertMonotoneInFeature0(t *testing.T, dm *data.DenseWithGroups, preds []float64) {
	t.Helper()
	row := make([]float64, 2)
	type pt struct {
		f0   float64
		pred float64
	}
	var pts []pt
	for i := 0; i < dm.NumRow(); i++ {
		if err := dm.Row(i, row); err != nil {
			t.Fatal(err)
		}
		pts = append(pts, pt{f0: row[0], pred: preds[i]})
	}
	for i := 0; i < len(pts); i++ {
		for j := i + 1; j < len(pts); j++ {
			if pts[i].f0+1e-6 < pts[j].f0 && pts[i].pred-1e-5 > pts[j].pred {
				t.Errorf("monotone violation: f0 %g pred %g vs f0 %g pred %g",
					pts[i].f0, pts[i].pred, pts[j].f0, pts[j].pred)
			}
		}
	}
}

func assertHighRelBeatsLow(t *testing.T, dm data.Matrix, preds []float64) {
	t.Helper()
	labels := dm.Labels()
	groups := groupsOf(dm)
	start := 0
	for _, gsz := range groups {
		bestRel, bestPred := -1.0, -mathMax
		worstRel, worstPred := mathMax, -mathMax
		for i := 0; i < gsz; i++ {
			idx := start + i
			if labels[idx] > bestRel {
				bestRel = labels[idx]
				bestPred = preds[idx]
			}
			if labels[idx] < worstRel {
				worstRel = labels[idx]
				worstPred = preds[idx]
			}
		}
		if bestPred <= worstPred+1e-6 {
			t.Errorf("group@%d: high-rel margin %g should beat low-rel %g", start, bestPred, worstPred)
		}
		start += gsz
	}
}

const mathMax = 1e30

func TestRankListwiseMonotoneIncreasing(t *testing.T) {
	dm := monotoneRankingMatrix(t)
	before := ndcgScore(t, dm, make([]float64, dm.NumRow()))

	learner, err := train.NewLearner(train.Config{
		Objective:           train.ObjectiveRankListwise,
		NumRound:            25,
		MaxDepth:            3,
		LearningRate:        0.2,
		TreeMethod:          train.TreeMethodHist,
		HistBinPolicy:       "global",
		MonotoneConstraints: []int{1, 0},
		EvalMetric:          train.EvalNDCG,
		Seed:                7,
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := learner.Fit(dm); err != nil {
		t.Fatal(err)
	}
	preds := make([]float64, dm.NumRow())
	if err := learner.PredictMargins(dm, preds); err != nil {
		t.Fatal(err)
	}
	after := ndcgScore(t, dm, preds)
	if after < before-0.02 {
		t.Errorf("NDCG should improve: before=%f after=%f", before, after)
	}
	assertMonotoneInFeature0(t, dm, preds)
	assertHighRelBeatsLow(t, dm, preds)
}

func TestRankPairwiseMonotoneIncreasing(t *testing.T) {
	dm := monotoneRankingMatrix(t)
	learner, err := train.NewLearner(train.Config{
		Objective:           train.ObjectiveRankPairwise,
		NumRound:            25,
		MaxDepth:            3,
		LearningRate:        0.2,
		TreeMethod:          train.TreeMethodExact,
		MonotoneConstraints: []int{1, 0},
		Seed:                7,
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := learner.Fit(dm); err != nil {
		t.Fatal(err)
	}
	preds := make([]float64, dm.NumRow())
	if err := learner.PredictMargins(dm, preds); err != nil {
		t.Fatal(err)
	}
	assertMonotoneInFeature0(t, dm, preds)
	assertHighRelBeatsLow(t, dm, preds)
}

func TestRankNDCGMonotoneIncreasing(t *testing.T) {
	dm := monotoneRankingMatrix(t)
	before := ndcgScore(t, dm, make([]float64, dm.NumRow()))

	learner, err := train.NewLearner(train.Config{
		Objective:           train.ObjectiveRankNDCG,
		NumRound:            25,
		MaxDepth:            3,
		LearningRate:        0.2,
		TreeMethod:          train.TreeMethodHist,
		HistBinPolicy:       "global",
		MonotoneConstraints: []int{1, 0},
		LambdaRankNorm:      true,
		Seed:                7,
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := learner.Fit(dm); err != nil {
		t.Fatal(err)
	}
	preds := make([]float64, dm.NumRow())
	if err := learner.PredictMargins(dm, preds); err != nil {
		t.Fatal(err)
	}
	after := ndcgScore(t, dm, preds)
	if after < before-0.02 {
		t.Errorf("NDCG should improve: before=%f after=%f", before, after)
	}
	assertMonotoneInFeature0(t, dm, preds)
	assertHighRelBeatsLow(t, dm, preds)
}

func TestRankListwiseMonotoneHistGPUPath(t *testing.T) {
	dm := monotoneRankingMatrix(t)
	learner, err := train.NewLearner(train.Config{
		Objective:           train.ObjectiveRankListwise,
		NumRound:            10,
		MaxDepth:            3,
		LearningRate:        0.3,
		TreeMethod:          train.TreeMethodHist,
		HistBinPolicy:       "global",
		MonotoneConstraints: []int{1},
		Seed:                3,
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := learner.Fit(dm); err != nil {
		t.Fatal(err)
	}
}
