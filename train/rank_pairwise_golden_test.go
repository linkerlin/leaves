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

type pairwiseGradGolden struct {
	Groups                []int     `json:"groups"`
	Labels                []float64 `json:"labels"`
	XGBCustomMarginRound1 []float64 `json:"xgb_custom_margin_round1"`
	Tolerance             struct {
		Margin float64 `json:"margin"`
	} `json:"tolerance"`
}

func loadPairwiseGradGoldenTrain(t *testing.T) pairwiseGradGolden {
	t.Helper()
	path := filepath.Join("..", "testdata", "rank_pairwise_grad_golden.json")
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read golden: %v", err)
	}
	var g pairwiseGradGolden
	if err := json.Unmarshal(b, &g); err != nil {
		t.Fatal(err)
	}
	return g
}

// TestRankPairwiseOneRoundMarginVsXGBGolden 全配对 grad 训练 1 轮 margin 对齐 XGBoost custom objective。
func TestRankPairwiseOneRoundMarginVsXGBGolden(t *testing.T) {
	g := loadPairwiseGradGoldenTrain(t)
	n := len(g.Labels)
	nfeat := 2
	feat := [][]float64{
		{1.0, 0.0}, {0.5, 0.1}, {0.0, 1.0},
		{0.9, 0.0}, {0.4, 0.2},
		{2.0, 0.0}, {1.2, 0.3}, {0.8, 0.5}, {0.1, 0.9},
	}
	vals := make([]float64, n*nfeat)
	for i, row := range feat {
		vals[i*nfeat] = row[0]
		vals[i*nfeat+1] = row[1]
	}
	dense, err := data.NewDense(vals, n, nfeat, g.Labels, nil)
	if err != nil {
		t.Fatal(err)
	}
	dm, err := data.NewDenseWithGroups(dense, g.Groups)
	if err != nil {
		t.Fatal(err)
	}

	learner, err := train.NewLearner(train.Config{
		Objective:            train.ObjectiveRankPairwise,
		NumRound:             1,
		MaxDepth:             6,
		LearningRate:         1.0,
		Lambda:               0,
		Gamma:                0,
		MinHessian:           0,
		TreeMethod:           train.TreeMethodExact,
		Seed:                 42,
		LambdaRankPairMethod: train.LambdaRankPairFull, // golden 为全配对公式
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

	tol := g.Tolerance.Margin
	if tol <= 0 {
		tol = 1e-4
	}
	if len(g.XGBCustomMarginRound1) != n {
		t.Fatalf("golden margin len %d != %d", len(g.XGBCustomMarginRound1), n)
	}
	for i := range preds {
		if math.Abs(preds[i]-g.XGBCustomMarginRound1[i]) > tol {
			t.Errorf("margin[%d]: leaves=%g xgb_golden=%g", i, preds[i], g.XGBCustomMarginRound1[i])
		}
	}
}
