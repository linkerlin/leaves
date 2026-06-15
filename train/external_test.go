package train

import (
	"testing"

	"github.com/dmitryikh/leaves/data"
)

func TestFitExternalBatchedHist(t *testing.T) {
	vals := []float64{
		0, 0, 1, 0, 0, 1, 1, 1,
		1, 0, 0, 1, 1, 0, 0, 1,
		0, 1, 1, 0, 1, 0, 0, 1,
		1, 1, 0, 0, 0, 1, 1, 0,
	}
	labels := []float64{0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0}
	dm, err := data.NewDense(vals, 16, 2, labels, nil)
	if err != nil {
		t.Fatal(err)
	}
	bm, err := data.SplitDense(dm, 4)
	if err != nil {
		t.Fatal(err)
	}
	if bm.NumBatches() != 4 {
		t.Fatalf("batches=%d", bm.NumBatches())
	}

	cfg := Config{
		Objective:    ObjectiveBinaryLogistic,
		NumRound:     3,
		MaxDepth:     3,
		LearningRate: 0.3,
		TreeMethod:   TreeMethodHist,
		HistBinPolicy: "global",
	}
	learner, err := FitExternal(cfg, bm)
	if err != nil {
		t.Fatal(err)
	}
	if learner.Model() == nil || len(learner.Model().Forest.Trees) == 0 {
		t.Fatal("expected trees from external hist training")
	}
}

func TestFitExternalExactMaterializes(t *testing.T) {
	vals := []float64{0, 0, 1, 1, 0, 1, 1, 0}
	labels := []float64{0, 1, 0, 1}
	dm, _ := data.NewDense(vals, 4, 2, labels, nil)
	bm, err := data.SplitDense(dm, 2)
	if err != nil {
		t.Fatal(err)
	}
	cfg := Config{
		Objective:    ObjectiveBinaryLogistic,
		NumRound:     2,
		MaxDepth:     2,
		TreeMethod:   TreeMethodExact,
	}
	learner, err := FitExternal(cfg, bm)
	if err != nil {
		t.Fatal(err)
	}
	if len(learner.Model().Forest.Trees) == 0 {
		t.Fatal("expected trees")
	}
}
