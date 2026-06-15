package io_test

import (
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	_ "github.com/linkerlin/leaves"
	"github.com/linkerlin/leaves/io"
)

func TestDetectFormatExtended(t *testing.T) {
	cases := []struct {
		file   string
		format io.Format
		errOK  bool
	}{
		{"sk_gradient_boosting_classifier.model", io.FormatSklearn, false},
		{"xgboost_smoke.json", io.FormatXGBoostJSON, false},
		{"breast_cancer_train.tsv", io.FormatUnknown, true},
	}
	for _, c := range cases {
		c := c
		t.Run(filepath.Base(c.file), func(t *testing.T) {
			// breast_cancer is .tsv not .txt - copy to temp .txt for detectTextFormat
			path := filepath.Join("..", "testdata", c.file)
			if c.errOK {
				dir := t.TempDir()
				path = filepath.Join(dir, "data.txt")
				b, err := os.ReadFile(filepath.Join("..", "testdata", c.file))
				if err != nil {
					t.Fatal(err)
				}
				if err := os.WriteFile(path, b, 0644); err != nil {
					t.Fatal(err)
				}
			}
			f, err := io.DetectFormat(path)
			if c.errOK {
				if err == nil {
					t.Fatalf("expected error for %s, got format %v", path, f)
				}
				if !strings.Contains(err.Error(), "tabular training data") {
					t.Fatalf("unexpected error: %v", err)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if f != c.format {
				t.Fatalf("format=%v want %v", f, c.format)
			}
		})
	}
}

func TestLoadFromFileDefaultAutoTransform(t *testing.T) {
	path := filepath.Join("..", "testdata", "xgboost_smoke.json")
	auto, err := io.LoadFromFile(path, io.DefaultLoadOptions())
	if err != nil {
		t.Fatal(err)
	}
	raw, err := io.LoadFromFile(path, &io.LoadOptions{AutoTransform: false})
	if err != nil {
		t.Fatal(err)
	}
	x := make([]float64, auto.NFeatures())
	pAuto := auto.PredictSingle(x, 0)
	pRaw := raw.PredictSingle(x, 0)
	if pAuto <= 0 || pAuto >= 1 {
		t.Fatalf("AutoTransform prob=%f want in (0,1)", pAuto)
	}
	if math.Abs(pAuto-pRaw) < 1e-6 {
		t.Fatalf("auto %f and raw %f should differ for binary:logistic", pAuto, pRaw)
	}
}
