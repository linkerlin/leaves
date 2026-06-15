package rankutil_test

import (
	"path/filepath"
	"testing"

	"github.com/linkerlin/leaves/data"
	"github.com/linkerlin/leaves/demos/movielens/rankutil"
	"github.com/linkerlin/leaves/train"
)

func TestNDCGAtKSmoke(t *testing.T) {
	path := filepath.Join("..", "..", "..", "testdata", "rank_smoke_train.tsv")
	dm, err := data.LoadRankingTSV(path, "\t")
	if err != nil {
		t.Fatal(err)
	}
	learner, err := train.NewLearner(train.Config{
		Objective:    train.ObjectiveRankNDCG,
		NumRound:     5,
		MaxDepth:     3,
		LearningRate: 0.3,
		NDCGK:        10,
		TreeMethod:   train.TreeMethodExact,
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := learner.Fit(dm); err != nil {
		t.Fatal(err)
	}
	preds, err := rankutil.PredictMargins(learner, dm)
	if err != nil {
		t.Fatal(err)
	}
	ndcg, err := rankutil.NDCGAtK(dm, preds, 10)
	if err != nil {
		t.Fatal(err)
	}
	if ndcg <= 0 || ndcg > 1 {
		t.Fatalf("NDCG@10=%f out of range", ndcg)
	}
}

func TestLoadXGBBaselineMissing(t *testing.T) {
	_, err := rankutil.LoadXGBBaseline(filepath.Join(t.TempDir(), "missing.json"))
	if err == nil {
		t.Fatal("expected error for missing baseline")
	}
}
