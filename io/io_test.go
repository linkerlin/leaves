package io_test

import (
	"math"
	"path/filepath"
	"testing"

	_ "github.com/linkerlin/leaves" // 注册加载器
	"github.com/linkerlin/leaves/io"
)

func TestLoadFromFileLightGBM(t *testing.T) {
	path := filepath.Join("..", "testdata", "lg_breast_cancer.txt")
	m, err := io.LoadFromFile(path, io.DefaultLoadOptions())
	if err != nil {
		t.Fatalf("LoadFromFile: %v", err)
	}
	if m.NFeatures() <= 0 {
		t.Fatal("expected positive feature count")
	}
	if m.NEstimators() <= 0 {
		t.Fatal("expected positive estimator count")
	}
}

func TestLoadFromFileXGBoost(t *testing.T) {
	path := filepath.Join("..", "testdata", "xgagaricus.model")
	m, err := io.LoadFromFile(path, io.DefaultLoadOptions())
	if err != nil {
		t.Fatalf("LoadFromFile: %v", err)
	}
	pred := m.PredictSingle(make([]float64, m.NFeatures()), 0)
	if math.IsNaN(pred) {
		t.Fatal("unexpected NaN prediction")
	}
}

func TestDetectFormat(t *testing.T) {
	cases := []struct {
		file   string
		format io.Format
	}{
		{"../testdata/lg_breast_cancer.txt", io.FormatLightGBM},
		{"../testdata/xgagaricus.model", io.FormatXGBoost},
		{"../testdata/xgboost_smoke.json", io.FormatXGBoostJSON},
		{"../testdata/xgboost_smoke.ubj", io.FormatXGBoostUBJSON},
	}
	for _, c := range cases {
		f, err := io.DetectFormat(c.file)
		if err != nil {
			t.Fatalf("%s: %v", c.file, err)
		}
		if f != c.format {
			t.Errorf("%s: expected %v, got %v", c.file, c.format, f)
		}
	}
}
