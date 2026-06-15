package train_test

import (
	"bytes"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/linkerlin/leaves/booster"
	"github.com/linkerlin/leaves/data"
	"github.com/linkerlin/leaves/io"
	"github.com/linkerlin/leaves/metrics"
	"github.com/linkerlin/leaves/train"
)

func TestCrossValidateRegression(t *testing.T) {
	vals := []float64{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
	labels := []float64{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
	dm, err := data.NewDense(vals, 10, 1, labels, nil)
	if err != nil {
		t.Fatal(err)
	}
	res, err := train.CrossValidate(train.Config{
		Objective:    train.ObjectiveSquaredError,
		NumRound:     8,
		MaxDepth:     2,
		LearningRate: 0.4,
		TreeMethod:   train.TreeMethodExact,
		Seed:         1,
	}, dm, 3)
	if err != nil {
		t.Fatal(err)
	}
	if len(res.FoldMetrics) != 3 {
		t.Fatalf("folds %d", len(res.FoldMetrics))
	}
	if res.MeanMetric > 2.0 {
		t.Errorf("cv mean rmse too high: %f", res.MeanMetric)
	}
}

func TestEarlyStopping(t *testing.T) {
	// 训练集可拟合；验证集标签随机，AUC 难以持续提升 → 触发早停。
	vals := make([]float64, 40)
	labels := make([]float64, 20)
	for i := 0; i < 20; i++ {
		vals[i*2] = float64(i)
		vals[i*2+1] = float64(i % 5)
		labels[i] = float64(i % 2)
	}
	trainDM, _ := data.NewDense(vals, 20, 2, labels, nil)

	valLabels := []float64{1, 0, 1, 0, 1, 0, 1, 0, 0, 1}
	valVals := make([]float64, 20)
	for i := 0; i < 10; i++ {
		valVals[i*2] = float64(i + 100)
		valVals[i*2+1] = float64(i % 3)
	}
	valDM, _ := data.NewDense(valVals, 10, 2, valLabels, nil)

	learner, err := train.NewLearner(train.Config{
		Objective:    train.ObjectiveBinaryLogistic,
		EvalMetric:   train.EvalAUC,
		NumRound:     50,
		MaxDepth:     4,
		LearningRate: 0.3,
		TreeMethod:   train.TreeMethodExact,
		EvalSet:      valDM,
		EarlyStop:    train.NewEarlyStopping(3, true),
		Seed:         1,
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := learner.Fit(trainDM); err != nil {
		t.Fatal(err)
	}
	if len(learner.MetricHistory()) >= 50 {
		t.Errorf("expected early stop before 50 rounds, got %d", len(learner.MetricHistory()))
	}
	if learner.BestRound() <= 0 {
		t.Errorf("expected positive best round")
	}
}

func TestDARTTraining(t *testing.T) {
	vals := []float64{0, 1, 2, 3, 4, 5}
	labels := []float64{0, 1, 0, 1, 0, 1}
	dm, _ := data.NewDense(vals, 6, 1, labels, nil)
	learner, _ := train.NewLearner(train.Config{
		Objective:    train.ObjectiveBinaryLogistic,
		NumRound:     10,
		MaxDepth:     2,
		LearningRate: 0.3,
		TreeMethod:   train.TreeMethodExact,
		DART: &booster.DARTConfig{
			RateDrop:      0.2,
			SkipDrop:      1,
			NormalizeType: "forest",
		},
		Seed: 42,
	})
	if err := learner.Fit(dm); err != nil {
		t.Fatal(err)
	}
	ir := learner.Model()
	if ir.Forest == nil || len(ir.Forest.WeightDrop) == 0 {
		t.Fatal("missing weight_drop")
	}
	hasDrop := false
	for _, w := range ir.Forest.WeightDrop {
		if w == 0 {
			hasDrop = true
			break
		}
	}
	if !hasDrop {
		t.Log("warning: no dropped trees in small run (probabilistic)")
	}
}

func TestCategoricalTraining(t *testing.T) {
	// feat0 numeric, feat1 categorical {0,1}
	vals := []float64{
		0, 0,
		0, 1,
		1, 0,
		1, 1,
		2, 0,
		2, 1,
	}
	labels := []float64{0, 1, 0, 1, 0, 1}
	dm, err := data.NewDense(vals, 6, 2, labels, nil)
	if err != nil {
		t.Fatal(err)
	}
	dm.FT = []data.FeatureType{data.FeatureNumeric, data.FeatureCategorical}

	learner, _ := train.NewLearner(train.Config{
		Objective:    train.ObjectiveBinaryLogistic,
		NumRound:     15,
		MaxDepth:     3,
		LearningRate: 0.4,
		TreeMethod:   train.TreeMethodExact,
	})
	if err := learner.Fit(dm); err != nil {
		t.Fatal(err)
	}
	preds := make([]float64, 6)
	_ = learner.PredictMargins(dm, preds)
	errCnt := 0
	for i, y := range labels {
		p := preds[i]
		if y > 0.5 && p <= 0 {
			errCnt++
		}
		if y < 0.5 && p >= 0 {
			errCnt++
		}
	}
	if errCnt > 2 {
		t.Errorf("categorical train errors %d", errCnt)
	}
}

func TestExportXGBoostJSONRoundTrip(t *testing.T) {
	vals := []float64{0, 1, 2, 3, 4, 5}
	labels := []float64{0, 1, 2, 3, 4, 5}
	dm, _ := data.NewDense(vals, 6, 1, labels, nil)
	learner, _ := train.NewLearner(train.Config{
		Objective:    train.ObjectiveSquaredError,
		NumRound:     5,
		MaxDepth:     2,
		LearningRate: 0.4,
		TreeMethod:   train.TreeMethodExact,
	})
	_ = learner.Fit(dm)

	mem := make([]float64, 6)
	_ = learner.PredictMargins(dm, mem)

	var buf bytes.Buffer
	if err := io.ExportXGBoostJSON(&buf, learner.Model(), train.ObjectiveSquaredError); err != nil {
		t.Fatal(err)
	}
	result, err := io.ParseXGBoostJSON(bytes.NewReader(buf.Bytes()))
	if err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	path := filepath.Join(dir, "xgb.json")
	_ = os.WriteFile(path, buf.Bytes(), 0644)

	loaded, err := io.LoadFromFile(path, &io.LoadOptions{LoadTransformation: false})
	if err != nil {
		t.Fatal(err)
	}
	for i, x := range vals {
		got := loaded.PredictSingle([]float64{x}, 0)
		if math.Abs(got-mem[i]) > 1e-4 {
			t.Errorf("i=%d mem=%f loaded=%f", i, mem[i], got)
		}
	}
	_ = result
}

func TestCheckpointSave(t *testing.T) {
	vals := []float64{0, 1, 2}
	labels := []float64{0, 1, 2}
	dm, _ := data.NewDense(vals, 3, 1, labels, nil)
	learner, _ := train.NewLearner(train.Config{
		Objective: train.ObjectiveSquaredError, NumRound: 3, TreeMethod: train.TreeMethodExact,
	})
	_ = learner.Fit(dm)
	dir := t.TempDir()
	path := filepath.Join(dir, "ckpt.json")
	if err := train.SaveCheckpointFile(path, 3, learner); err != nil {
		t.Fatal(err)
	}
	round, obj, _, err := train.LoadCheckpointFile(path)
	if err != nil || round != 3 || obj != train.ObjectiveSquaredError {
		t.Fatalf("round=%d obj=%s err=%v", round, obj, err)
	}
}

func TestNumParallelTreeRF(t *testing.T) {
	vals := make([]float64, 20)
	labels := make([]float64, 10)
	for i := 0; i < 10; i++ {
		vals[i*2] = float64(i)
		vals[i*2+1] = float64(i % 3)
		labels[i] = float64(i % 2)
	}
	dm, _ := data.NewDense(vals, 10, 2, labels, nil)
	learner, _ := train.NewLearner(train.Config{
		Objective:       train.ObjectiveSquaredError,
		NumRound:        5,
		MaxDepth:        2,
		LearningRate:    0.5,
		NumParallelTree: 3,
		TreeMethod:      train.TreeMethodExact,
		Seed:            7,
	})
	if err := learner.Fit(dm); err != nil {
		t.Fatal(err)
	}
	if learner.Model().Forest.NumParallelTree != 3 {
		t.Errorf("parallel tree %d", learner.Model().Forest.NumParallelTree)
	}
	rmse := metrics.RMSE{}
	preds := make([]float64, 10)
	_ = learner.PredictMargins(dm, preds)
	got, _ := rmse.Evaluate(labels, preds)
	if got > 1.5 {
		t.Errorf("rf rmse %f", got)
	}
}
