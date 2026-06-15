package io_test

import (
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	_ "github.com/linkerlin/leaves"
	"github.com/linkerlin/leaves/io"
	"github.com/linkerlin/leaves/predict"
)

func TestParseXGBoostJSONSmoke(t *testing.T) {
	path := filepath.Join("..", "testdata", "xgboost_smoke.json")
	result, err := io.ParseXGBoostJSONFile(path)
	if err != nil {
		t.Fatalf("ParseXGBoostJSONFile: %v", err)
	}
	if result.IR == nil || result.IR.Forest == nil {
		t.Fatal("expected forest in ModelIR")
	}
	if len(result.IR.Forest.Trees) != 5 {
		t.Errorf("expected 5 trees, got %d", len(result.IR.Forest.Trees))
	}
	if result.IR.Forest.NumFeatures != 8 {
		t.Errorf("expected 8 features, got %d", result.IR.Forest.NumFeatures)
	}
	if len(result.IR.Forest.IterationIndptr) != 6 {
		t.Errorf("expected iteration_indptr len 6, got %d", len(result.IR.Forest.IterationIndptr))
	}
}

func TestLoadFromFileXGBoostJSON(t *testing.T) {
	path := filepath.Join("..", "testdata", "xgboost_smoke.json")
	m, err := io.LoadFromFile(path, &io.LoadOptions{
		LoadTransformation: true,
		Backend:            io.BackendNative,
	})
	if err != nil {
		t.Fatalf("LoadFromFile: %v", err)
	}
	if m.NEstimators() != 5 {
		t.Errorf("NEstimators: expected 5, got %d", m.NEstimators())
	}

	predPath := filepath.Join("..", "testdata", "xgboost_smoke_pred.txt")
	wantStr, err := os.ReadFile(predPath)
	if err != nil {
		t.Fatalf("missing %s: %v", predPath, err)
	}
	want, err := strconv.ParseFloat(strings.TrimSpace(string(wantStr)), 64)
	if err != nil {
		t.Fatal(err)
	}
	got := m.PredictSingle(make([]float64, m.NFeatures()), 0)
	if math.Abs(got-want) > 1e-5 {
		t.Errorf("PredictSingle: got %f want %f (diff=%e)", got, want, math.Abs(got-want))
	}
}

func TestPredictRequestDense(t *testing.T) {
	path := filepath.Join("..", "testdata", "xgboost_smoke.json")
	m, err := io.LoadFromFile(path, &io.LoadOptions{LoadTransformation: true})
	if err != nil {
		t.Fatal(err)
	}
	vals := make([]float64, m.NFeatures()*2)
	out := make([]float64, 2)
	err = m.PredictWithRequest(predict.Request{
		Matrix: predict.DenseMatrix{Values: vals, Rows: 2, Cols: m.NFeatures()},
		Output: predict.OutputValue,
	}, out)
	if err != nil {
		t.Fatal(err)
	}
	if math.IsNaN(out[0]) || math.IsNaN(out[1]) {
		t.Fatal("NaN predictions")
	}
}

func TestOutputMarginXGBoostJSON(t *testing.T) {
	path := filepath.Join("..", "testdata", "xgboost_smoke.json")
	m, err := io.LoadFromFile(path, &io.LoadOptions{LoadTransformation: true})
	if err != nil {
		t.Fatal(err)
	}
	vals := make([]float64, m.NFeatures())
	out := make([]float64, 1)
	err = m.PredictWithRequest(predict.Request{
		Matrix: predict.DenseMatrix{Values: vals, Rows: 1, Cols: m.NFeatures()},
		Output: predict.OutputMargin,
	}, out)
	if err != nil {
		t.Fatal(err)
	}
	want := -2.122276
	if math.Abs(out[0]-want) > 1e-4 {
		t.Errorf("margin: got %f want %f", out[0], want)
	}
}
