package train_test

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/dmitryikh/leaves/data"
	"github.com/dmitryikh/leaves/train"
)

// xgbTopKMarginGolden 来自 testdata/gen_rank_pairwise_grad.py 同数据集 + XGBoost 3.x topk 配置。
var xgbTopKMarginGolden = []float64{
	1.0, 0.0, -1.0, 1.0, -1.0, 1.0, 0.3333333432674408, -0.3333333432674408, -1.0,
}

// TestRankPairwiseTopKVsXGBoostRound1 对标 XGBoost lambdarank_pair_method=topk（默认 k=32）。
func TestRankPairwiseTopKVsXGBoostRound1(t *testing.T) {
	path := filepath.Join("..", "testdata", "rank_pairwise_grad_golden.json")
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var golden struct {
		Labels []float64 `json:"labels"`
		Groups []int     `json:"groups"`
	}
	if err := json.Unmarshal(b, &golden); err != nil {
		t.Fatal(err)
	}

	n := len(golden.Labels)
	feat := [][]float64{
		{1.0, 0.0}, {0.5, 0.1}, {0.0, 1.0},
		{0.9, 0.0}, {0.4, 0.2},
		{2.0, 0.0}, {1.2, 0.3}, {0.8, 0.5}, {0.1, 0.9},
	}
	vals := make([]float64, n*2)
	for i, row := range feat {
		vals[i*2] = row[0]
		vals[i*2+1] = row[1]
	}
	dense, _ := data.NewDense(vals, n, 2, golden.Labels, nil)
	dm, _ := data.NewDenseWithGroups(dense, golden.Groups)

	learner, err := train.NewLearner(train.Config{
		Objective:                  train.ObjectiveRankPairwise,
		NumRound:                   1,
		MaxDepth:                   6,
		LearningRate:               1.0,
		Lambda:                     0,
		Gamma:                      0,
		MinHessian:                 0,
		TreeMethod:                 train.TreeMethodExact,
		Seed:                       42,
		LambdaRankPairMethod:       train.LambdaRankPairTopK,
		LambdaRankNumPairPerSample: 32,
		LambdaRankNormalization:    true,
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := learner.Fit(dm); err != nil {
		t.Fatal(err)
	}
	preds := make([]float64, n)
	if err := learner.PredictMargins(dm, preds); err != nil {
		t.Fatal(err)
	}
	for i := range preds {
		if math.Abs(preds[i]-xgbTopKMarginGolden[i]) > 1e-4 {
			t.Errorf("margin[%d]: leaves=%g xgb=%g", i, preds[i], xgbTopKMarginGolden[i])
		}
	}
}

// TestRankPairwiseMeanTrains mean 配对可训练（XGBoost mean 采样 RNG 因平台而异，不做 round-1 margin 硬对齐）。
func TestRankPairwiseMeanTrains(t *testing.T) {
	path := filepath.Join("..", "testdata", "rank_pairwise_grad_golden.json")
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var golden struct {
		Labels []float64 `json:"labels"`
		Groups []int     `json:"groups"`
	}
	if err := json.Unmarshal(b, &golden); err != nil {
		t.Fatal(err)
	}

	n := len(golden.Labels)
	feat := [][]float64{
		{1.0, 0.0}, {0.5, 0.1}, {0.0, 1.0},
		{0.9, 0.0}, {0.4, 0.2},
		{2.0, 0.0}, {1.2, 0.3}, {0.8, 0.5}, {0.1, 0.9},
	}
	vals := make([]float64, n*2)
	for i, row := range feat {
		vals[i*2] = row[0]
		vals[i*2+1] = row[1]
	}
	dense, _ := data.NewDense(vals, n, 2, golden.Labels, nil)
	dm, _ := data.NewDenseWithGroups(dense, golden.Groups)

	learner, err := train.NewLearner(train.Config{
		Objective:                  train.ObjectiveRankPairwise,
		NumRound:                   1,
		MaxDepth:                   6,
		LearningRate:               1.0,
		Lambda:                     0,
		Gamma:                      0,
		MinHessian:                 0,
		TreeMethod:                 train.TreeMethodExact,
		Seed:                       42,
		LambdaRankPairMethod:       train.LambdaRankPairMean,
		LambdaRankNumPairPerSample: 1,
		LambdaRankNormalization:    true,
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := learner.Fit(dm); err != nil {
		t.Fatal(err)
	}
	preds := make([]float64, n)
	if err := learner.PredictMargins(dm, preds); err != nil {
		t.Fatal(err)
	}
	for i, p := range preds {
		if math.IsNaN(p) || math.IsInf(p, 0) {
			t.Fatalf("margin[%d] invalid %v", i, p)
		}
	}
	if preds[0] <= preds[2] {
		t.Errorf("group0: high-rel margin %g should beat low-rel %g", preds[0], preds[2])
	}
}
