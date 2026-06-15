package train_test

import (
	"path/filepath"
	"testing"

	"github.com/linkerlin/leaves/data"
	"github.com/linkerlin/leaves/train"
)

func TestResumeFit(t *testing.T) {
	vals := []float64{0, 1, 2, 3, 4, 5}
	labels := []float64{0, 1, 2}
	dm, err := data.NewDense(vals, 3, 2, labels, nil)
	if err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	ckpt := filepath.Join(dir, "ckpt.json")
	cfg := train.Config{
		Objective:      train.ObjectiveSquaredError,
		NumRound:       2,
		MaxDepth:       2,
		TreeMethod:     train.TreeMethodExact,
		CheckpointPath: ckpt,
		CheckpointEvery: 1,
	}
	learner, err := train.NewLearner(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := learner.Fit(dm); err != nil {
		t.Fatal(err)
	}
	treesAfter2 := len(learner.Model().Forest.Trees)

	resumed, err := train.ResumeFit(ckpt, train.Config{
		Objective:  train.ObjectiveSquaredError,
		NumRound:   4,
		MaxDepth:   2,
		TreeMethod: train.TreeMethodExact,
	}, dm)
	if err != nil {
		t.Fatal(err)
	}
	if n := len(resumed.Model().Forest.Trees); n <= treesAfter2 {
		t.Fatalf("expected more trees after resume: before=%d after=%d", treesAfter2, n)
	}
}
