package train_test

import (
	"math"
	"os"
	"path/filepath"
	"testing"

	_ "github.com/linkerlin/leaves"
	"github.com/linkerlin/leaves/data"
	"github.com/linkerlin/leaves/io"
	"github.com/linkerlin/leaves/mat"
	"github.com/linkerlin/leaves/metrics"
	"github.com/linkerlin/leaves/model"
	"github.com/linkerlin/leaves/train"
	"github.com/linkerlin/leaves/tree"
)

func TestFitGBLinearSynthetic(t *testing.T) {
	vals := []float64{0, 1, 2, 3, 4, 5}
	labels := []float64{1, 3, 5, 7, 9, 11} // y = 2x + 1
	dm, err := data.NewDense(vals, 6, 1, labels, nil)
	if err != nil {
		t.Fatal(err)
	}
	learner, err := train.NewLearner(train.Config{
		Booster:      train.BoosterGBLinear,
		Objective:    train.ObjectiveSquaredError,
		NumRound:     30,
		LearningRate: 0.5,
		Lambda:       0.1,
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := learner.Fit(dm); err != nil {
		t.Fatal(err)
	}
	preds := make([]float64, 6)
	_ = learner.PredictMargins(dm, preds)
	rmse := metrics.RMSE{}
	got, _ := rmse.Evaluate(labels, preds)
	if got > 0.5 {
		t.Errorf("gblinear RMSE too high: %f", got)
	}

	dir := t.TempDir()
	path := filepath.Join(dir, "lin.leaves.json")
	if err := learner.Save(path); err != nil {
		t.Fatal(err)
	}
	loaded, err := io.LoadFromFile(path, &io.LoadOptions{LoadTransformation: false})
	if err != nil {
		t.Fatal(err)
	}
	for i, x := range vals {
		got := loaded.PredictSingle([]float64{x}, 0)
		if math.Abs(got-preds[i]) > 1e-5 {
			t.Errorf("i=%d roundtrip %f vs %f", i, got, preds[i])
		}
	}
}

func TestFitMulticlassSynthetic(t *testing.T) {
	path := filepath.Join("..", "testdata", "multiclass_test.tsv")
	if _, err := os.Stat(path); err != nil {
		t.Skip("missing multiclass_test.tsv")
	}
	dm, err := data.LoadDenseTSVLabelFirst(path)
	if err != nil {
		t.Fatal(err)
	}
	learner, err := train.NewLearner(train.Config{
		Objective:    train.ObjectiveMultiSoftmax,
		NumClass:     5,
		NumRound:     25,
		MaxDepth:     4,
		LearningRate: 0.2,
		TreeMethod:   train.TreeMethodHist,
		EvalMetric:   train.EvalMLogLoss,
		Seed:         7,
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := learner.Fit(dm); err != nil {
		t.Fatal(err)
	}

	n := dm.NumRow()
	margins := make([]float64, n*5)
	if err := learner.PredictMargins(dm, margins); err != nil {
		t.Fatal(err)
	}
	labels := dm.Labels()
	errCnt := 0
	for i := 0; i < n; i++ {
		best, bestC := -1e18, -1
		for c := 0; c < 5; c++ {
			v := margins[i*5+c]
			if v > best {
				best = v
				bestC = c
			}
		}
		if float64(bestC) != labels[i] {
			errCnt++
		}
	}
	errRate := float64(errCnt) / float64(n)
	if errRate > 0.35 {
		t.Errorf("multiclass error rate too high: %f", errRate)
	}
}

func TestFitPoissonSynthetic(t *testing.T) {
	// y ~ Poisson(exp(0.5*x))
	vals := []float64{0, 1, 2, 3, 4, 5}
	labels := []float64{1, 1, 2, 3, 5, 8}
	dm, err := data.NewDense(vals, 6, 1, labels, nil)
	if err != nil {
		t.Fatal(err)
	}
	learner, err := train.NewLearner(train.Config{
		Objective:    train.ObjectivePoisson,
		NumRound:     20,
		MaxDepth:     2,
		LearningRate: 0.3,
		TreeMethod:   train.TreeMethodExact,
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := learner.Fit(dm); err != nil {
		t.Fatal(err)
	}
	margins := make([]float64, 6)
	_ = learner.PredictMargins(dm, margins)
	for i, y := range labels {
		mu := math.Exp(margins[i])
		if mu < 0 {
			t.Fatalf("negative mu at %d", i)
		}
		if y > 0 && mu < 0.01 {
			t.Errorf("sample %d mu too small %f", i, mu)
		}
	}
}

func TestFitCSRRegression(t *testing.T) {
	vals := []float64{0, 1, 2, 3, 4, 5}
	labels := []float64{0, 1, 2, 3, 4, 5}
	dense, err := data.NewDense(vals, 6, 1, labels, nil)
	if err != nil {
		t.Fatal(err)
	}
	csrMat, err := mat.CSRMatFromArray(vals, 6, 1)
	if err != nil {
		t.Fatal(err)
	}
	dm, err := data.NewCSR(csrMat.RowHeaders, csrMat.ColIndexes, csrMat.Values, 1, labels, nil)
	if err != nil {
		t.Fatal(err)
	}

	learner, err := train.NewLearner(train.Config{
		Objective:    train.ObjectiveSquaredError,
		NumRound:     12,
		MaxDepth:     2,
		LearningRate: 0.4,
		TreeMethod:   train.TreeMethodExact,
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := learner.Fit(dm); err != nil {
		t.Fatal(err)
	}

	densePred := make([]float64, 6)
	csrPred := make([]float64, 6)
	_ = learner.PredictMargins(dense, densePred)

	learner2, _ := train.NewLearner(train.Config{
		Objective: train.ObjectiveSquaredError, NumRound: 12, MaxDepth: 2, LearningRate: 0.4, TreeMethod: train.TreeMethodExact, Seed: 0,
	})
	_ = learner2.Fit(dm)
	_ = learner2.PredictMargins(dm, csrPred)

	rmse := metrics.RMSE{}
	got, _ := rmse.Evaluate(labels, csrPred)
	if got > 0.5 {
		t.Errorf("CSR RMSE %f", got)
	}
}

func TestFitSampleWeights(t *testing.T) {
	vals := []float64{0, 1, 2, 3}
	labels := []float64{0, 2, 100, 102} // outliers at end
	weights := []float64{1, 1, 0.01, 0.01}
	dm, err := data.NewDense(vals, 4, 1, labels, weights)
	if err != nil {
		t.Fatal(err)
	}
	learner, err := train.NewLearner(train.Config{
		Objective:    train.ObjectiveSquaredError,
		NumRound:     15,
		MaxDepth:     2,
		LearningRate: 0.4,
		TreeMethod:   train.TreeMethodExact,
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := learner.Fit(dm); err != nil {
		t.Fatal(err)
	}
	preds := make([]float64, 4)
	_ = learner.PredictMargins(dm, preds)
	if math.Abs(preds[0]-0) > 1.0 || math.Abs(preds[1]-2) > 1.0 {
		t.Errorf("weighted fit should follow heavy samples: %v", preds)
	}
}

func TestFitSubsampleColsample(t *testing.T) {
	vals := make([]float64, 40)
	labels := make([]float64, 20)
	for i := 0; i < 20; i++ {
		vals[i*2] = float64(i)
		vals[i*2+1] = float64(20 - i)
		labels[i] = vals[i*2] - vals[i*2+1]
	}
	dm, err := data.NewDense(vals, 20, 2, labels, nil)
	if err != nil {
		t.Fatal(err)
	}
	learner, err := train.NewLearner(train.Config{
		Objective:       train.ObjectiveSquaredError,
		NumRound:        15,
		MaxDepth:        3,
		LearningRate:    0.4,
		TreeMethod:      train.TreeMethodExact,
		Subsample:       0.8,
		ColsampleByTree: 0.8,
		Seed:            99,
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := learner.Fit(dm); err != nil {
		t.Fatal(err)
	}
	preds := make([]float64, 20)
	_ = learner.PredictMargins(dm, preds)
	rmse := metrics.RMSE{}
	got, _ := rmse.Evaluate(labels, preds)
	if got > 5.0 {
		t.Errorf("subsample RMSE %f", got)
	}
}

func TestGBLinearEngineParity(t *testing.T) {
	vals := []float64{1, 2, 3}
	labels := []float64{3, 5, 7}
	dm, _ := data.NewDense(vals, 3, 1, labels, nil)
	learner, _ := train.NewLearner(train.Config{
		Booster: train.BoosterGBLinear, Objective: train.ObjectiveSquaredError,
		NumRound: 20, LearningRate: 0.5,
	})
	_ = learner.Fit(dm)

	mem := make([]float64, 3)
	_ = learner.PredictMargins(dm, mem)

	ir := learner.Model()
	eng, err := model.NewEngine(ir, tree.ApplyTransformRaw, tree.TransformRaw, tree.BackendNative)
	if err != nil {
		t.Fatal(err)
	}
	for i, x := range vals {
		var p [1]float64
		_ = eng.Predict([]float64{x}, 0, p[:])
		if math.Abs(p[0]-mem[i]) > 1e-6 {
			t.Errorf("i=%d eng %f mem %f", i, p[0], mem[i])
		}
	}
}
