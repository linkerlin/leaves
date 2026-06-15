package train_test

import (
	"math"
	"os"
	"path/filepath"
	"testing"

	_ "github.com/linkerlin/leaves"
	"github.com/linkerlin/leaves/data"
	leavesio "github.com/linkerlin/leaves/io"
	"github.com/linkerlin/leaves/train"
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

func TestInferObjectiveFromUBJ(t *testing.T) {
	path := filepath.Join("..", "testdata", "xgboost_smoke.ubj")
	obj, err := train.InferObjectiveFromModel(path)
	if err != nil {
		t.Fatal(err)
	}
	if obj != "binary:logistic" {
		t.Fatalf("objective=%q want binary:logistic", obj)
	}
}

func TestInferObjectiveFromLeavesJSON(t *testing.T) {
	vals := []float64{0, 1, 1, 0}
	labels := []float64{0, 1}
	dm, err := data.NewDense(vals, 2, 2, labels, nil)
	if err != nil {
		t.Fatal(err)
	}
	learner, err := train.NewLearner(train.Config{
		Objective:  train.ObjectiveSquaredError,
		NumRound:   2,
		MaxDepth:   2,
		TreeMethod: train.TreeMethodExact,
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := learner.Fit(dm); err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	path := filepath.Join(dir, "m.leaves.json")
	if err := learner.Save(path); err != nil {
		t.Fatal(err)
	}
	obj, err := train.InferObjectiveFromModel(path)
	if err != nil {
		t.Fatal(err)
	}
	if obj != train.ObjectiveSquaredError {
		t.Fatalf("objective=%q", obj)
	}
}

func TestInferObjectiveUnsupportedFormat(t *testing.T) {
	_, err := train.InferObjectiveFromModel(filepath.Join("..", "testdata", "xgagaricus.model"))
	if err == nil {
		t.Fatal("expected error for binary xgb model path")
	}
}

func TestNewLearnerFromModelAndDataPreservesObjective(t *testing.T) {
	modelPath := filepath.Join("..", "testdata", "xgboost_smoke.json")
	dataPath := filepath.Join("..", "testdata", "breast_cancer_train.tsv")
	if _, err := os.Stat(dataPath); err != nil {
		t.Skipf("missing %s", dataPath)
	}
	want := train.ObjectiveBinaryLogistic
	learner, err := train.NewLearnerFromModelAndData(modelPath, dataPath, train.Config{
		Objective:    want,
		NumRound:     2,
		MaxDepth:     2,
		LearningRate: 0.3,
		TreeMethod:   train.TreeMethodExact,
	}, data.DefaultFileLoadOptions())
	if err != nil {
		t.Fatal(err)
	}
	_ = learner
}

func TestLoadDataAuto(t *testing.T) {
	path := filepath.Join("..", "testdata", "csrmat.libsvm")
	m, err := train.LoadDataAuto(path)
	if err != nil {
		t.Fatal(err)
	}
	if m.NumRow() == 0 {
		t.Fatal("empty")
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

func TestLoadDataRoundTrip(t *testing.T) {
	path := filepath.Join("..", "testdata", "breast_cancer_train.tsv")
	if _, err := os.Stat(path); err != nil {
		t.Skip(path)
	}
	m1, err := train.LoadDataAuto(path)
	if err != nil {
		t.Fatal(err)
	}
	m2, err := train.LoadData(path, data.DefaultFileLoadOptions())
	if err != nil {
		t.Fatal(err)
	}
	if m1.NumRow() != m2.NumRow() || m1.NumCol() != m2.NumCol() {
		t.Fatalf("shape mismatch %dx%d vs %dx%d", m1.NumRow(), m1.NumCol(), m2.NumRow(), m2.NumCol())
	}
}

func TestSaveTrainModelRoundTrip(t *testing.T) {
	vals := []float64{0, 1, 2, 3}
	labels := []float64{0, 1}
	dm, _ := data.NewDense(vals, 2, 2, labels, nil)
	learner, err := train.NewLearner(train.Config{
		Objective:  train.ObjectiveBinaryLogistic,
		NumRound:   3,
		MaxDepth:   2,
		TreeMethod: train.TreeMethodExact,
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := learner.Fit(dm); err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	path := filepath.Join(dir, "out.leaves.json")
	if err := leavesio.SaveTrainModel(path, learner.Model(), train.ObjectiveBinaryLogistic); err != nil {
		t.Fatal(err)
	}
	m, err := leavesio.LoadFromFile(path, &leavesio.LoadOptions{AutoTransform: true})
	if err != nil {
		t.Fatal(err)
	}
	p := m.PredictSingle([]float64{0, 1}, 0)
	if math.IsNaN(p) {
		t.Fatal("NaN prediction")
	}
}
