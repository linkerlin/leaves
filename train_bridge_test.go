package leaves_test

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/linkerlin/leaves"
	"github.com/linkerlin/leaves/data"
)

func TestRootDefaultLoadOptionsAutoTransform(t *testing.T) {
	opts := leaves.DefaultLoadOptions()
	if !opts.AutoTransform {
		t.Fatal("root DefaultLoadOptions should enable AutoTransform")
	}
}

func TestRootLoadDataAuto(t *testing.T) {
	path := filepath.Join("testdata", "csrmat.libsvm")
	if _, err := os.Stat(path); err != nil {
		t.Skip(path)
	}
	m, err := leaves.LoadDataAuto(path)
	if err != nil {
		t.Fatal(err)
	}
	if m.NumRow() == 0 {
		t.Fatal("empty")
	}
}

func TestRootInferObjectiveFromModel(t *testing.T) {
	path := filepath.Join("testdata", "xgboost_smoke.json")
	obj, err := leaves.InferObjectiveFromModel(path)
	if err != nil {
		t.Fatal(err)
	}
	if obj != "binary:logistic" {
		t.Fatalf("objective=%q", obj)
	}
}

func TestTrainBridgeNewLearner(t *testing.T) {
	vals := []float64{0, 1, 1, 0, 0, 1}
	labels := []float64{0, 1, 1}
	dm, err := data.NewDense(vals, 3, 2, labels, nil)
	if err != nil {
		t.Fatal(err)
	}
	learner, err := leaves.NewLearner(leaves.TrainConfig{
		Objective:  leaves.TrainObjectiveSquaredError,
		NumRound:   2,
		MaxDepth:   2,
		TreeMethod: leaves.TrainTreeMethodExact,
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := learner.Fit(dm); err != nil {
		t.Fatal(err)
	}
	if learner.Model() == nil {
		t.Fatal("nil model")
	}
}
