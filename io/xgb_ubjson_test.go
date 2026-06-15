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

func TestParseXGBoostUBJSONSmoke(t *testing.T) {
	path := filepath.Join("..", "testdata", "xgboost_smoke.ubj")
	result, err := io.ParseXGBoostUBJSONFile(path)
	if err != nil {
		t.Fatalf("ParseXGBoostUBJSONFile: %v", err)
	}
	if result.IR == nil || result.IR.Forest == nil {
		t.Fatal("expected forest in ModelIR")
	}
	if len(result.IR.Forest.Trees) != 5 {
		t.Errorf("expected 5 trees, got %d", len(result.IR.Forest.Trees))
	}
	if result.Objective != "binary:logistic" {
		t.Errorf("objective: got %q", result.Objective)
	}
}

func TestLoadFromFileXGBoostUBJSON(t *testing.T) {
	path := filepath.Join("..", "testdata", "xgboost_smoke.ubj")
	m, err := io.LoadFromFile(path, &io.LoadOptions{
		LoadTransformation: true,
		Backend:            io.BackendNative,
	})
	if err != nil {
		t.Fatalf("LoadFromFile: %v", err)
	}

	predPath := filepath.Join("..", "testdata", "xgboost_smoke_ubj_pred.txt")
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

func TestDetectFormatUBJSON(t *testing.T) {
	path := filepath.Join("..", "testdata", "xgboost_smoke.ubj")
	format, err := io.DetectFormat(path)
	if err != nil {
		t.Fatal(err)
	}
	if format != io.FormatXGBoostUBJSON {
		t.Errorf("format: got %v want FormatXGBoostUBJSON", format)
	}
}

func TestUBJSONMatchesJSONPrediction(t *testing.T) {
	jsonPath := filepath.Join("..", "testdata", "xgboost_smoke.json")
	ubjPath := filepath.Join("..", "testdata", "xgboost_smoke.ubj")
	opts := &io.LoadOptions{LoadTransformation: true}

	mJSON, err := io.LoadFromFile(jsonPath, opts)
	if err != nil {
		t.Fatal(err)
	}
	mUBJ, err := io.LoadFromFile(ubjPath, opts)
	if err != nil {
		t.Fatal(err)
	}

	vals := make([]float64, mJSON.NFeatures())
	outJSON := make([]float64, 1)
	outUBJ := make([]float64, 1)
	req := predict.Request{
		Matrix: predict.DenseMatrix{Values: vals, Rows: 1, Cols: mJSON.NFeatures()},
		Output: predict.OutputValue,
	}
	if err := mJSON.PredictWithRequest(req, outJSON); err != nil {
		t.Fatal(err)
	}
	if err := mUBJ.PredictWithRequest(req, outUBJ); err != nil {
		t.Fatal(err)
	}
	if math.Abs(outJSON[0]-outUBJ[0]) > 1e-9 {
		t.Errorf("JSON vs UBJSON: %f != %f", outJSON[0], outUBJ[0])
	}
}
