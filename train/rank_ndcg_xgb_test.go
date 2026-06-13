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

type ndcgMarginGolden struct {
	Labels []float64 `json:"labels"`
	Groups []int     `json:"groups"`
	TopK   struct {
		NumPairPerSample  int       `json:"num_pair_per_sample"`
		XGBMarginRound1   []float64 `json:"xgb_margin_round1"`
	} `json:"topk"`
	Mean struct {
		XGBMarginRound1 []float64 `json:"xgb_margin_round1"`
	} `json:"mean"`
	Tolerance struct {
		Margin float64 `json:"margin"`
	} `json:"tolerance"`
}

func loadNDCGMarginGolden(t *testing.T) ndcgMarginGolden {
	t.Helper()
	path := filepath.Join("..", "testdata", "rank_ndcg_grad_golden.json")
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var g ndcgMarginGolden
	if err := json.Unmarshal(b, &g); err != nil {
		t.Fatal(err)
	}
	return g
}

func ndcgGoldenMatrix(t *testing.T, g ndcgMarginGolden) *data.DenseWithGroups {
	t.Helper()
	n := len(g.Labels)
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
	dense, err := data.NewDense(vals, n, 2, g.Labels, nil)
	if err != nil {
		t.Fatal(err)
	}
	dm, err := data.NewDenseWithGroups(dense, g.Groups)
	if err != nil {
		t.Fatal(err)
	}
	return dm
}

func TestRankNDCGTopKVsXGBoostRound1(t *testing.T) {
	g := loadNDCGMarginGolden(t)
	dm := ndcgGoldenMatrix(t, g)
	learner, err := train.NewLearner(train.Config{
		Objective:                  train.ObjectiveRankNDCG,
		NumRound:                   1,
		MaxDepth:                   6,
		LearningRate:               1.0,
		Lambda:                     0,
		Gamma:                      0,
		MinHessian:                 0,
		TreeMethod:                 train.TreeMethodExact,
		Seed:                       42,
		LambdaRankNorm:             true,
		LambdaRankPairMethod:       train.LambdaRankPairTopK,
		LambdaRankNumPairPerSample: g.TopK.NumPairPerSample,
		LambdaRankNormalization:    true,
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
	tol := g.Tolerance.Margin
	if tol <= 0 {
		tol = 1e-4
	}
	want := g.TopK.XGBMarginRound1
	for i := range preds {
		if math.Abs(preds[i]-want[i]) > tol {
			t.Errorf("margin[%d]: leaves=%g xgb=%g", i, preds[i], want[i])
		}
	}
}

func TestRankNDCGMeanTrains(t *testing.T) {
	g := loadNDCGMarginGolden(t)
	dm := ndcgGoldenMatrix(t, g)
	learner, err := train.NewLearner(train.Config{
		Objective:                  train.ObjectiveRankNDCG,
		NumRound:                   1,
		MaxDepth:                   6,
		LearningRate:               1.0,
		Lambda:                     0,
		TreeMethod:                 train.TreeMethodExact,
		Seed:                       42,
		LambdaRankNorm:             true,
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
	preds := make([]float64, dm.NumRow())
	if err := learner.PredictMargins(dm, preds); err != nil {
		t.Fatal(err)
	}
	for i, p := range preds {
		if math.IsNaN(p) || math.IsInf(p, 0) {
			t.Fatalf("margin[%d] invalid %v", i, p)
		}
	}
}
