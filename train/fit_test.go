package train_test

import (
	"math"
	"os"
	"path/filepath"
	"testing"

	_ "github.com/linkerlin/leaves"
	"github.com/linkerlin/leaves/data"
	"github.com/linkerlin/leaves/io"
	"github.com/linkerlin/leaves/metrics"
	"github.com/linkerlin/leaves/model"
	"github.com/linkerlin/leaves/train"
	"github.com/linkerlin/leaves/tree"
)

func TestFitRegressionSynthetic(t *testing.T) {
	// y ≈ 2*x0 - x1
	vals := []float64{
		0, 0,
		1, 0,
		0, 1,
		1, 1,
		2, 0,
		2, 2,
	}
	labels := []float64{0, 2, -1, 1, 4, 2}
	dm, err := data.NewDense(vals, 6, 2, labels, nil)
	if err != nil {
		t.Fatal(err)
	}

	learner, err := train.NewLearner(train.Config{
		Objective:    "reg:squarederror",
		NumRound:     20,
		MaxDepth:     3,
		LearningRate: 0.5,
		Lambda:       1.0,
		TreeMethod:   "exact",
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := learner.Fit(dm); err != nil {
		t.Fatal(err)
	}

	before := make([]float64, 6)
	learner.PredictMargins(dm, before)

	ir := learner.Model()
	eng, err := model.NewEngine(ir, tree.ApplyTransformRaw, tree.TransformRaw, tree.BackendNative)
	if err != nil {
		t.Fatal(err)
	}
	after := make([]float64, 6)
	row := make([]float64, 2)
	for i := 0; i < 6; i++ {
		_ = dm.Row(i, row)
		_ = eng.Predict(row, 0, after[i:i+1])
	}
	for i := range before {
		if math.Abs(before[i]-after[i]) > 1e-9 {
			t.Errorf("sample %d: memory %f engine %f", i, before[i], after[i])
		}
	}

	rmse := metrics.RMSE{}
	got, _ := rmse.Evaluate(labels, after)
	if got > 0.5 {
		t.Errorf("RMSE too high after fit: %f", got)
	}
}

func TestFitSaveLoadRoundTrip(t *testing.T) {
	vals := []float64{0, 1, 2, 3, 4, 5}
	labels := []float64{0, 1, 2, 3, 4, 5}
	dm, err := data.NewDense(vals, 6, 1, labels, nil)
	if err != nil {
		t.Fatal(err)
	}

	learner, err := train.NewLearner(train.Config{
		Objective:    "reg:squarederror",
		NumRound:     15,
		MaxDepth:     2,
		LearningRate: 0.4,
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := learner.Fit(dm); err != nil {
		t.Fatal(err)
	}

	mem := make([]float64, 6)
	learner.PredictMargins(dm, mem)

	dir := t.TempDir()
	path := filepath.Join(dir, "model.leaves.json")
	if err := io.SaveLeavesJSONFile(path, learner.Model(), "reg:squarederror"); err != nil {
		t.Fatal(err)
	}

	loaded, err := io.LoadFromFile(path, &io.LoadOptions{LoadTransformation: false})
	if err != nil {
		t.Fatal(err)
	}

	filePreds := make([]float64, 6)
	for i := 0; i < 6; i++ {
		x := []float64{vals[i]}
		filePreds[i] = loaded.PredictSingle(x, 0)
	}

	for i := range mem {
		if math.Abs(mem[i]-filePreds[i]) > 1e-6 {
			t.Errorf("sample %d: mem %f file %f", i, mem[i], filePreds[i])
		}
	}
	_ = os.Remove(path)
}

func TestFitBinaryLogistic(t *testing.T) {
	vals := []float64{0.1, 0.9, 0.2, 0.8, 0.15, 0.85}
	labels := []float64{0, 1, 0, 1, 0, 1}
	dm, err := data.NewDense(vals, 6, 1, labels, nil)
	if err != nil {
		t.Fatal(err)
	}

	learner, err := train.NewLearner(train.Config{
		Objective:    "binary:logistic",
		NumRound:     10,
		MaxDepth:     2,
		LearningRate: 0.5,
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := learner.Fit(dm); err != nil {
		t.Fatal(err)
	}

	margins := make([]float64, 6)
	learner.PredictMargins(dm, margins)
	for i, y := range labels {
		if y > 0.5 && margins[i] <= 0 {
			t.Errorf("sample %d: expected positive margin, got %f", i, margins[i])
		}
		if y < 0.5 && margins[i] >= 0 {
			t.Errorf("sample %d: expected negative margin, got %f", i, margins[i])
		}
	}
}
