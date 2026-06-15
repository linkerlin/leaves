package io_test

import (
	"bytes"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/dmitryikh/leaves/io"
	"github.com/dmitryikh/leaves/model"
)

func TestParseXGBoostJSONGblinear(t *testing.T) {
	path := filepath.Join("..", "testdata", "xgboost_gblinear_smoke.json")
	if _, err := os.Stat(path); err != nil {
		t.Skip("run: cd testdata && python gen_xgb_gblinear.py")
	}
	result, err := io.ParseXGBoostJSONFile(path)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if result.IR.Linear == nil || result.IR.Forest != nil {
		t.Fatalf("expected Linear ModelIR, got Kind=%v", result.IR.Kind)
	}
	if len(result.IR.Linear.Weights) == 0 {
		t.Fatal("empty weights")
	}
}

func TestLoadXGBoostJSONGblinearPredict(t *testing.T) {
	modelPath := filepath.Join("..", "testdata", "xgboost_gblinear_smoke.json")
	predPath := filepath.Join("..", "testdata", "xgboost_gblinear_smoke_pred.txt")
	if _, err := os.Stat(modelPath); err != nil {
		t.Skip("run: cd testdata && python gen_xgb_gblinear.py")
	}
	m, err := io.LoadFromFile(modelPath, &io.LoadOptions{LoadTransformation: true})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()

	wantStr, err := os.ReadFile(predPath)
	if err != nil {
		t.Fatal(err)
	}
	want, err := strconv.ParseFloat(strings.TrimSpace(string(wantStr)), 64)
	if err != nil {
		t.Fatal(err)
	}
	got := m.PredictSingle(make([]float64, m.NFeatures()), 0)
	if math.Abs(got-want) > 1e-4 {
		t.Errorf("PredictSingle: got %f want %f", got, want)
	}
}

func TestLoadXGBoostJSONGammaTransform(t *testing.T) {
	modelPath := filepath.Join("..", "testdata", "xgboost_gamma_smoke.json")
	predPath := filepath.Join("..", "testdata", "xgboost_gamma_smoke_pred.txt")
	if _, err := os.Stat(modelPath); err != nil {
		t.Skip("run: cd testdata && python gen_xgb_gamma.py")
	}
	m, err := io.LoadFromFile(modelPath, &io.LoadOptions{LoadTransformation: true})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()

	wantStr, err := os.ReadFile(predPath)
	if err != nil {
		t.Fatal(err)
	}
	want, err := strconv.ParseFloat(strings.TrimSpace(string(wantStr)), 64)
	if err != nil {
		t.Fatal(err)
	}
	got := m.PredictSingle(make([]float64, m.NFeatures()), 0)
	if math.Abs(got-want) > 1e-3 {
		t.Errorf("gamma PredictSingle: got %f want %f", got, want)
	}

	mRaw, err := io.LoadFromFile(modelPath, &io.LoadOptions{LoadTransformation: false})
	if err != nil {
		t.Fatal(err)
	}
	defer mRaw.Close()
	raw := mRaw.PredictSingle(make([]float64, m.NFeatures()), 0)
	if math.Abs(math.Exp(raw)-want) > 1e-2 {
		t.Errorf("exp(margin): exp(%f)=%f want %f", raw, math.Exp(raw), want)
	}
}

func TestXGBoostCategoricalExportRoundTrip(t *testing.T) {
	path := filepath.Join("..", "testdata", "xgboost_categorical_smoke.json")
	result, err := io.ParseXGBoostJSONFile(path)
	if err != nil {
		t.Skip(err)
	}
	if len(result.IR.Forest.XGBCatsRaw) == 0 {
		t.Fatal("expected XGBCatsRaw preserved")
	}

	var buf bytes.Buffer
	if err := io.ExportXGBoostJSON(&buf, result.IR, result.Objective); err != nil {
		t.Fatal(err)
	}
	reloaded, err := io.ParseXGBoostJSON(bytes.NewReader(buf.Bytes()))
	if err != nil {
		t.Fatal(err)
	}

	outType, transform := io.ObjectiveToTransform(reloaded.Objective, false)
	reloaded.IR.NOutputGroups = io.NOutputGroupsForTransform(reloaded.IR.NRawOutputGroups, outType)
	mExport, err := model.NewEnsembleFromIR(reloaded.IR, transform, outType, io.BackendNative)
	if err != nil {
		t.Fatal(err)
	}
	defer mExport.Close()

	mOrig, err := io.LoadFromFile(path, &io.LoadOptions{LoadTransformation: false})
	if err != nil {
		t.Fatal(err)
	}
	defer mOrig.Close()

	fvals := []float64{0.5, 1.0}
	p0 := mOrig.PredictSingle(fvals, 0)
	p1 := mExport.PredictSingle(fvals, 0)
	if math.Abs(p0-p1) > 1e-5 {
		t.Errorf("categorical round-trip predict: orig=%f export=%f", p0, p1)
	}

	var doc map[string]interface{}
	if err := json.Unmarshal(buf.Bytes(), &doc); err != nil {
		t.Fatal(err)
	}
	modelObj := doc["learner"].(map[string]interface{})["gradient_booster"].(map[string]interface{})["model"].(map[string]interface{})
	if _, ok := modelObj["cats"]; !ok {
		t.Fatal("exported model missing cats")
	}
}

func TestLoadXGBoostBinaryAgaricus(t *testing.T) {
	path := filepath.Join("..", "testdata", "xgagaricus.model")
	if _, err := os.Stat(path); err != nil {
		t.Skip("missing agaricus model")
	}
	m, err := io.LoadFromFile(path, &io.LoadOptions{LoadTransformation: true})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()
	if m.NEstimators() <= 0 {
		t.Fatal("expected trees")
	}
	got := m.PredictSingle(make([]float64, m.NFeatures()), 0)
	if math.IsNaN(got) || got < 0 || got > 1 {
		t.Errorf("unexpected prediction: %f", got)
	}
}

func TestLoadXGBoostJSONMulticlassSoftprob(t *testing.T) {
	modelPath := filepath.Join("..", "testdata", "xgboost_multiclass_smoke.json")
	predPath := filepath.Join("..", "testdata", "xgboost_multiclass_smoke_pred.txt")
	if _, err := os.Stat(predPath); err != nil {
		t.Skip("run: cd testdata && python gen_xgb_multiclass_pred.py")
	}
	m, err := io.LoadFromFile(modelPath, &io.LoadOptions{LoadTransformation: true})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	if m.NOutputGroups() != 3 {
		t.Fatalf("NOutputGroups: got %d want 3", m.NOutputGroups())
	}
	wantLines, err := os.ReadFile(predPath)
	if err != nil {
		t.Fatal(err)
	}
	var wants []float64
	for _, line := range strings.Split(strings.TrimSpace(string(wantLines)), "\n") {
		v, err := strconv.ParseFloat(strings.TrimSpace(line), 64)
		if err != nil {
			t.Fatal(err)
		}
		wants = append(wants, v)
	}
	got := make([]float64, 3)
	if err := m.Predict(make([]float64, m.NFeatures()), 0, got); err != nil {
		t.Fatal(err)
	}
	for i := range wants {
		if math.Abs(got[i]-wants[i]) > 1e-3 {
			t.Errorf("class %d: got %f want %f", i, got[i], wants[i])
		}
	}
}

func testXGBJSONPredGolden(t *testing.T, modelName, predName string, tol float64) {
	modelPath := filepath.Join("..", "testdata", modelName)
	predPath := filepath.Join("..", "testdata", predName)
	if _, err := os.Stat(modelPath); err != nil {
		t.Skip("missing " + modelName)
	}
	m, err := io.LoadFromFile(modelPath, &io.LoadOptions{LoadTransformation: true})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	wantStr, err := os.ReadFile(predPath)
	if err != nil {
		t.Skip("missing " + predName)
	}
	want, err := strconv.ParseFloat(strings.TrimSpace(string(wantStr)), 64)
	if err != nil {
		t.Fatal(err)
	}
	got := m.PredictSingle(make([]float64, m.NFeatures()), 0)
	if math.Abs(got-want) > tol {
		t.Errorf("%s: got %f want %f", modelName, got, want)
	}
}

func TestLoadXGBoostJSONPoisson(t *testing.T) {
	testXGBJSONPredGolden(t, "xgboost_poisson_smoke.json", "xgboost_poisson_smoke_pred.txt", 1e-2)
}

func TestLoadXGBoostJSONDart(t *testing.T) {
	modelPath := filepath.Join("..", "testdata", "xgboost_dart_smoke.json")
	if _, err := os.Stat(modelPath); err != nil {
		t.Skip("run: cd testdata && python gen_xgb_dart.py")
	}
	result, err := io.ParseXGBoostJSONFile(modelPath)
	if err != nil {
		t.Fatal(err)
	}
	if result.IR.Kind != model.KindDART {
		t.Fatalf("kind: %v", result.IR.Kind)
	}
	if len(result.IR.Forest.WeightDrop) == 0 {
		t.Fatal("missing weight_drop")
	}
	testXGBJSONPredGolden(t, "xgboost_dart_smoke.json", "xgboost_dart_smoke_pred.txt", 1e-4)
}

func TestXGBoostJSONPreservesVersionAndBoostFromAverage(t *testing.T) {
	path := filepath.Join("..", "testdata", "xgboost_multiclass_smoke.json")
	result, err := io.ParseXGBoostJSONFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(result.IR.XGBVersion) < 2 || result.IR.XGBVersion[0] != 3 {
		t.Fatalf("version: %v", result.IR.XGBVersion)
	}
	if !result.IR.XGBBoostFromAverage {
		t.Fatal("expected boost_from_average=true")
	}
}

