package objective_test

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/dmitryikh/leaves/objective"
)

type ndcgGradGolden struct {
	Groups []int     `json:"groups"`
	Labels []float64 `json:"labels"`
	Preds  []float64 `json:"preds"`
	TopK   struct {
		NumPairPerSample int       `json:"num_pair_per_sample"`
		Grad             []float64 `json:"grad"`
	} `json:"topk"`
	Mean struct {
		Grad []float64 `json:"grad"`
	} `json:"mean"`
	Tolerance struct {
		Grad float64 `json:"grad"`
	} `json:"tolerance"`
}

func loadNDCGGradGolden(t *testing.T) ndcgGradGolden {
	t.Helper()
	path := filepath.Join("..", "testdata", "rank_ndcg_grad_golden.json")
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read golden (run testdata/gen_rank_ndcg_grad.py): %v", err)
	}
	var g ndcgGradGolden
	if err := json.Unmarshal(b, &g); err != nil {
		t.Fatal(err)
	}
	return g
}

func TestRankNDCGGradGoldenTopK(t *testing.T) {
	g := loadNDCGGradGolden(t)
	testNDCGGradGolden(t, g, g.TopK.Grad, objective.RankPairTopK, g.TopK.NumPairPerSample)
}

func TestRankNDCGGradGoldenMeanNonZero(t *testing.T) {
	g := loadNDCGGradGolden(t)
	n := len(g.Labels)
	grad := make([]float64, n)
	hess := make([]float64, n)
	obj := objective.NewRankNDCG(objective.RankTrainConfig{
		LambdaNorm:          true,
		PairMethod:          objective.RankPairMean,
		NumPairPerSample:    1,
		PairSeed:            42,
		LambdaNormalization: true,
	})
	dm := rankDM{rows: n, labels: g.Labels}
	if err := objective.GradHessRanking(obj, dm, g.Groups, g.Preds, grad, hess); err != nil {
		t.Fatal(err)
	}
	var sum float64
	for _, v := range grad {
		sum += math.Abs(v)
	}
	if sum < 1e-9 {
		t.Fatal("expected non-zero mean ndcg grad")
	}
}

func testNDCGGradGolden(t *testing.T, g ndcgGradGolden, want []float64, method objective.RankPairMethod, npp int) {
	t.Helper()
	n := len(g.Labels)
	grad := make([]float64, n)
	hess := make([]float64, n)
	obj := objective.NewRankNDCG(objective.RankTrainConfig{
		LambdaNorm:           true,
		PairMethod:           method,
		NumPairPerSample:     npp,
		PairSeed:             42,
		LambdaNormalization:  true,
	})
	dm := rankDM{rows: n, labels: g.Labels}
	if err := objective.GradHessRanking(obj, dm, g.Groups, g.Preds, grad, hess); err != nil {
		t.Fatal(err)
	}
	tol := g.Tolerance.Grad
	if tol <= 0 {
		tol = 1e-10
	}
	for i := range grad {
		if math.Abs(grad[i]-want[i]) > tol {
			t.Errorf("grad[%d]: got %g want %g", i, grad[i], want[i])
		}
	}
}
