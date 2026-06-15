package train_test

import (
	"encoding/json"
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
)

func TestT1BreastCancerE2E(t *testing.T) {
	trainPath := filepath.Join("..", "testdata", "breast_cancer_train.tsv")
	testPath := filepath.Join("..", "testdata", "breast_cancer_labeled_test.tsv")
	baselinePath := filepath.Join("..", "testdata", "breast_cancer_xgb_baseline.json")
	if _, err := os.Stat(trainPath); err != nil {
		t.Skipf("missing %s (run testdata/gen_breast_cancer_train.py)", trainPath)
	}

	trainDM, err := data.LoadDenseTSV(trainPath)
	if err != nil {
		t.Fatal(err)
	}
	testDM, err := data.LoadDenseTSV(testPath)
	if err != nil {
		t.Fatal(err)
	}

	learner, err := train.NewLearner(train.Config{
		Objective:    train.ObjectiveBinaryLogistic,
		NumRound:     50,
		MaxDepth:     4,
		LearningRate: 0.1,
		Lambda:       1.0,
		TreeMethod:   train.TreeMethodHist,
		EvalMetric:   train.EvalAUC,
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := learner.Fit(trainDM); err != nil {
		t.Fatal(err)
	}

	dir := t.TempDir()
	modelPath := filepath.Join(dir, "model.leaves.json")
	if err := learner.Save(modelPath); err != nil {
		t.Fatal(err)
	}

	memMargins := predictMargins(testDM, learner)
	loaded, err := io.LoadFromFile(modelPath, &io.LoadOptions{LoadTransformation: false})
	if err != nil {
		t.Fatal(err)
	}
	fileMargins := predictMarginsLoaded(testDM, loaded)

	for i := range memMargins {
		if math.Abs(memMargins[i]-fileMargins[i]) > 1e-6 {
			t.Fatalf("sample %d: mem %f file %f", i, memMargins[i], fileMargins[i])
		}
	}

	labels := testDM.Labels()
	auc := metrics.AUC{}
	gotAUC, err := auc.Evaluate(labels, memMargins)
	if err != nil {
		t.Fatal(err)
	}
	if gotAUC < 0.90 {
		t.Errorf("leaves AUC too low: %f (want >= 0.90)", gotAUC)
	}

	if baseline, err := loadBaselineAUC(baselinePath); err == nil {
		if math.Abs(gotAUC-baseline) > 0.05 {
			t.Errorf("AUC gap vs XGBoost baseline: leaves=%f xgb=%f (tolerance 0.05)", gotAUC, baseline)
		}
	}

	hist := learner.MetricHistory()
	if len(hist) != 50 {
		t.Errorf("metric history len %d, want 50", len(hist))
	}
}

func predictMargins(dm *data.Dense, learner *train.Learner) []float64 {
	n := dm.NumRow()
	out := make([]float64, n)
	if err := learner.PredictMargins(dm, out); err != nil {
		panic(err)
	}
	return out
}

func predictMarginsLoaded(dm *data.Dense, m *model.Ensemble) []float64 {
	n := dm.NumRow()
	row := make([]float64, dm.NumCol())
	out := make([]float64, n)
	for i := 0; i < n; i++ {
		_ = dm.Row(i, row)
		out[i] = m.PredictSingle(row, 0)
	}
	return out
}

func loadBaselineAUC(path string) (float64, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return 0, err
	}
	var doc struct {
		TestAUC float64 `json:"test_auc"`
	}
	if err := json.Unmarshal(b, &doc); err != nil {
		return 0, err
	}
	return doc.TestAUC, nil
}
