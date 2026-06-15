package io_test

import (
	"path/filepath"
	"testing"

	"github.com/linkerlin/leaves/io"
)

func TestXGBoostJSONSplitGainStats(t *testing.T) {
	path := filepath.Join("..", "testdata", "xgboost_smoke.json")
	result, err := io.ParseXGBoostJSONFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if result.IR == nil || result.IR.Forest == nil {
		t.Fatal("nil forest")
	}
	if len(result.IR.Forest.Trees) == 0 {
		t.Fatal("no trees")
	}
	t0 := &result.IR.Forest.Trees[0]
	if len(t0.SplitGain) == 0 {
		t.Fatal("expected SplitGain from JSON loss_changes")
	}
	if len(t0.SumHess) == 0 {
		t.Fatal("expected SumHess from JSON sum_hessian")
	}
	if t0.SplitGain[0] <= 0 {
		t.Errorf("expected positive SplitGain[0], got %f", t0.SplitGain[0])
	}
	if t0.SumHess[0] <= 0 {
		t.Errorf("expected positive SumHess[0], got %f", t0.SumHess[0])
	}

	m, err := io.LoadFromFile(path, &io.LoadOptions{LoadTransformation: false})
	if err != nil {
		t.Fatal(err)
	}
	f := m.Forest()
	if f == nil || len(f.Trees[0].SplitGain) == 0 {
		t.Fatal("LoadFromFile path should preserve SplitGain")
	}
}
