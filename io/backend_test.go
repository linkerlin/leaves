package io_test

import (
	"math"
	"path/filepath"
	"testing"

	_ "github.com/linkerlin/leaves/v2"
	"github.com/linkerlin/leaves/v2/io"
	"github.com/linkerlin/leaves/v2/linear"
	"github.com/linkerlin/leaves/v2/tree"
)

func TestSelectBackendFromIR(t *testing.T) {
	path := filepath.Join("..", "testdata", "xgboost_smoke.json")
	result, err := io.ParseXGBoostJSONFile(path)
	if err != nil {
		t.Fatalf("ParseXGBoostJSONFile: %v", err)
	}

	cases := []struct {
		name string
		hint tree.WorkloadHint
		want tree.Backend
	}{
		{
			name: "default small batch",
			hint: tree.DefaultWorkloadHint(),
			want: tree.BackendNative,
		},
		{
			name: "wasm numeric",
			hint: tree.WorkloadHint{Target: tree.DeployWASM},
			want: tree.BackendNative,
		},
		{
			name: "large batch no gpu → native (2.1 诚实化)",
			hint: tree.WorkloadHint{BatchSize: 512, HasGPU: false},
			want: tree.BackendNative,
		},
		{
			name: "mid batch cpu → native",
			hint: tree.WorkloadHint{BatchSize: 64},
			want: tree.BackendNative,
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := io.SelectBackend(result.IR, c.hint)
			if got != c.want {
				t.Errorf("SelectBackend: expected %v, got %v", c.want, got)
			}
		})
	}
	// large batch + GPU：2.1 起默认也 Native（显式/profiling 才走 Born）
	t.Run("large batch gpu", func(t *testing.T) {
		got := io.SelectBackend(result.IR, tree.WorkloadHint{BatchSize: 512, HasGPU: true})
		if got != tree.BackendNative {
			t.Errorf("want Native, got %v", got)
		}
	})
}

func TestLoadFromFileBackendAutoNative(t *testing.T) {
	path := filepath.Join("..", "testdata", "xgboost_smoke.json")
	opts := &io.LoadOptions{
		LoadTransformation: true,
		Backend:            io.BackendAuto,
		Workload:           tree.DefaultWorkloadHint(),
	}
	auto, err := io.LoadFromFile(path, opts)
	if err != nil {
		t.Fatalf("LoadFromFile auto: %v", err)
	}
	native, err := io.LoadFromFile(path, &io.LoadOptions{
		LoadTransformation: true,
		Backend:            io.BackendNative,
	})
	if err != nil {
		t.Fatalf("LoadFromFile native: %v", err)
	}

	if _, ok := auto.Engine().(*tree.NativeEngine); !ok {
		t.Fatalf("expected NativeEngine, got %T", auto.Engine())
	}

	fvals := make([]float64, auto.NFeatures())
	got := auto.PredictSingle(fvals, 0)
	want := native.PredictSingle(fvals, 0)
	if math.Abs(got-want) > 1e-9 {
		t.Errorf("predictions differ: auto=%v native=%v", got, want)
	}
}

func TestLoadFromFileBackendAutoWASM(t *testing.T) {
	path := filepath.Join("..", "testdata", "xgboost_smoke.json")
	opts := &io.LoadOptions{
		LoadTransformation: true,
		Backend:            io.BackendAuto,
		Workload:           tree.WorkloadHint{Target: tree.DeployWASM},
	}
	auto, err := io.LoadFromFile(path, opts)
	if err != nil {
		t.Fatalf("LoadFromFile auto: %v", err)
	}
	native, err := io.LoadFromFile(path, &io.LoadOptions{
		LoadTransformation: true,
		Backend:            io.BackendNative,
	})
	if err != nil {
		t.Fatalf("LoadFromFile native: %v", err)
	}

	if _, ok := auto.Engine().(*tree.NativeEngine); !ok {
		t.Fatalf("expected NativeEngine, got %T", auto.Engine())
	}

	fvals := make([]float64, auto.NFeatures())
	got := auto.PredictSingle(fvals, 0)
	want := native.PredictSingle(fvals, 0)
	if math.Abs(got-want) > 1e-5 {
		t.Errorf("predictions differ: auto=%v native=%v", got, want)
	}
}

func TestLoadFromFileBackendAutoLinear(t *testing.T) {
	path := filepath.Join("..", "testdata", "xgblin_agaricus.model")
	opts := &io.LoadOptions{
		LoadTransformation: true,
		Backend:            io.BackendAuto,
		Workload:           tree.WorkloadHint{BatchSize: 512, HasGPU: true, Target: tree.DeployWASM},
	}
	m, err := io.LoadFromFile(path, opts)
	if err != nil {
		t.Fatalf("LoadFromFile: %v", err)
	}
	if _, ok := m.Engine().(*linear.NativeEngine); !ok {
		t.Fatalf("linear model should use linear.NativeEngine, got %T", m.Engine())
	}
}
