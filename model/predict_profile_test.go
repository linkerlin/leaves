package model_test

import (
	"path/filepath"
	"testing"

	_ "github.com/dmitryikh/leaves"
	"github.com/dmitryikh/leaves/io"
	"github.com/dmitryikh/leaves/predict"
)

func TestPredictWithProfile(t *testing.T) {
	path := filepath.Join("..", "testdata", "xgboost_smoke.json")
	e, err := io.LoadFromFile(path, io.DefaultLoadOptions())
	if err != nil {
		t.Fatal(err)
	}
	feats := []float64{0, 1, 0, 1, 0, 1, 0, 1}
	out := make([]float64, 1)
	prof, err := e.PredictWithProfile(predict.Request{
		Matrix: predict.DenseMatrix{Values: feats, Rows: 1, Cols: len(feats)},
		Output: predict.OutputValue,
	}, out)
	if err != nil {
		t.Fatal(err)
	}
	if prof.Rows != 1 {
		t.Fatalf("profile %+v", prof)
	}
	if prof.Elapsed < 0 {
		t.Fatalf("negative elapsed")
	}
}
