package train_test

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/dmitryikh/leaves/data"
	"github.com/dmitryikh/leaves/train"
)

func TestInferObjectiveFromModel(t *testing.T) {
	path := filepath.Join("..", "testdata", "xgboost_smoke.json")
	obj, err := train.InferObjectiveFromModel(path)
	if err != nil {
		t.Fatal(err)
	}
	if obj != "binary:logistic" {
		t.Fatalf("objective=%q want binary:logistic", obj)
	}
}

func TestNewLearnerFromModelAndData(t *testing.T) {
	modelPath := filepath.Join("..", "testdata", "xgboost_smoke.json")
	dataPath := filepath.Join("..", "testdata", "breast_cancer_train.tsv")
	if _, err := os.Stat(dataPath); err != nil {
		t.Skipf("missing %s", dataPath)
	}

	learner, err := train.NewLearnerFromModelAndData(modelPath, dataPath, train.Config{
		NumRound:     5,
		MaxDepth:     3,
		LearningRate: 0.3,
		TreeMethod:   train.TreeMethodExact,
	}, data.DefaultFileLoadOptions())
	if err != nil {
		t.Fatal(err)
	}
	if learner == nil {
		t.Fatal("nil learner")
	}
	ir := learner.Model()
	if ir == nil || ir.Forest == nil || len(ir.Forest.Trees) == 0 {
		t.Fatal("expected trained model")
	}
}

func TestNewLearnerFromFileAuto(t *testing.T) {
	dataPath := filepath.Join("..", "testdata", "breast_cancer_train.tsv")
	if _, err := os.Stat(dataPath); err != nil {
		t.Skipf("missing %s", dataPath)
	}

	learner, err := train.NewLearnerFromFile(dataPath, train.Config{
		Objective:    train.ObjectiveBinaryLogistic,
		NumRound:     3,
		MaxDepth:     2,
		LearningRate: 0.3,
		TreeMethod:   train.TreeMethodExact,
	}, data.DefaultFileLoadOptions())
	if err != nil {
		t.Fatal(err)
	}
	if learner.Model() == nil {
		t.Fatal("nil model")
	}
}
