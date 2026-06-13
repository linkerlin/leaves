package objective_test

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/dmitryikh/leaves/objective"
)

type pairwiseGradGolden struct {
	Seed                     int     `json:"seed"`
	Groups                   []int   `json:"groups"`
	Labels                   []float64 `json:"labels"`
	Preds                    []float64 `json:"preds"`
	Grad                     []float64 `json:"grad"`
	Hess                     []float64 `json:"hess"`
	GroupPairs               []struct {
		Group int `json:"group"`
		Size  int `json:"size"`
		Pairs []struct {
			Hi       int     `json:"hi"`
			Lo       int     `json:"lo"`
			Rho      float64 `json:"rho"`
			Lambda   float64 `json:"lambda"`
			HessPair float64 `json:"hess_pair"`
		} `json:"pairs"`
	} `json:"group_pairs"`
	XGBCustomMarginRound1 []float64 `json:"xgb_custom_margin_round1"`
	Tolerance             struct {
		Grad   float64 `json:"grad"`
		Hess   float64 `json:"hess"`
		Margin float64 `json:"margin"`
	} `json:"tolerance"`
}

func loadPairwiseGradGolden(t *testing.T) pairwiseGradGolden {
	t.Helper()
	path := filepath.Join("..", "testdata", "rank_pairwise_grad_golden.json")
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read golden (run testdata/gen_rank_pairwise_grad.py): %v", err)
	}
	var g pairwiseGradGolden
	if err := json.Unmarshal(b, &g); err != nil {
		t.Fatal(err)
	}
	return g
}

func TestRankPairwiseGradGolden(t *testing.T) {
	g := loadPairwiseGradGolden(t)
	n := len(g.Labels)
	if len(g.Preds) != n || len(g.Grad) != n || len(g.Hess) != n {
		t.Fatal("golden length mismatch")
	}
	grad := make([]float64, n)
	hess := make([]float64, n)
	obj := objective.NewRankPairwise(objective.RankTrainConfig{})
	dm := rankDM{rows: n, labels: g.Labels}
	if err := objective.GradHessRanking(obj, dm, g.Groups, g.Preds, grad, hess); err != nil {
		t.Fatal(err)
	}
	tolG := g.Tolerance.Grad
	if tolG <= 0 {
		tolG = 1e-12
	}
	tolH := g.Tolerance.Hess
	if tolH <= 0 {
		tolH = 1e-12
	}
	for i := range grad {
		if math.Abs(grad[i]-g.Grad[i]) > tolG {
			t.Errorf("grad[%d]: leaves=%g golden=%g", i, grad[i], g.Grad[i])
		}
		if math.Abs(hess[i]-g.Hess[i]) > tolH {
			t.Errorf("hess[%d]: leaves=%g golden=%g", i, hess[i], g.Hess[i])
		}
	}
}

func TestRankPairwisePerPairLambdaGolden(t *testing.T) {
	g := loadPairwiseGradGolden(t)
	obj := objective.NewRankPairwise(objective.RankTrainConfig{})
	start := 0
	for _, gp := range g.GroupPairs {
		gsz := gp.Size
		preds := g.Preds[start : start+gsz]
		labels := g.Labels[start : start+gsz]
		for _, want := range gp.Pairs {
			got := objective.PairwiseLambdaAt(preds, labels, want.Hi, want.Lo)
			if math.Abs(got.Rho-want.Rho) > 1e-12 {
				t.Errorf("group %d pair (%d,%d) rho: got %g want %g", gp.Group, want.Hi, want.Lo, got.Rho, want.Rho)
			}
			if math.Abs(got.Lambda-want.Lambda) > 1e-12 {
				t.Errorf("group %d pair (%d,%d) lambda: got %g want %g", gp.Group, want.Hi, want.Lo, got.Lambda, want.Lambda)
			}
			if math.Abs(got.HessPair-want.HessPair) > 1e-12 {
				t.Errorf("group %d pair (%d,%d) hess_pair: got %g want %g", gp.Group, want.Hi, want.Lo, got.HessPair, want.HessPair)
			}
		}
		// 由逐对 λ 聚合应等于组梯度
		grad := make([]float64, gsz)
		hess := make([]float64, gsz)
		obj.GradHessGroup(preds, labels, nil, grad, hess)
		gstart := start
		for i := 0; i < gsz; i++ {
			if math.Abs(grad[i]-g.Grad[gstart+i]) > 1e-12 {
				t.Errorf("group %d grad[%d] aggregate mismatch", gp.Group, i)
			}
		}
		start += gsz
	}
}
