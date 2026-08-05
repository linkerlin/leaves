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
			want: tree.BackendBornCPU,
		},
		{
			name: "large batch no gpu → born cpu",
			hint: tree.WorkloadHint{BatchSize: 512, HasGPU: false},
			want: tree.BackendBornCPU,
		},
		{
			name: "mid batch cpu",
			hint: tree.WorkloadHint{BatchSize: 64},
			want: tree.BackendBornCPU,
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
	// large batch + GPU：有 WebGPU → BornGPU，否则 BornCPU
	t.Run("large batch gpu", func(t *testing.T) {
		got := io.SelectBackend(result.IR, tree.WorkloadHint{BatchSize: 512, HasGPU: true})
		if tree.BornWebGPUAvailable() {
			if got != tree.BackendBornGPU {
				t.Errorf("want BornGPU, got %v", got)
			}
		} else if got != tree.BackendBornCPU {
			t.Errorf("want BornCPU fallback, got %v", got)
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
	bornCPU, err := io.LoadFromFile(path, &io.LoadOptions{
		LoadTransformation: true,
		Backend:            io.BackendBornCPU,
	})
	if err != nil {
		t.Fatalf("LoadFromFile born cpu: %v", err)
	}

	if _, ok := auto.Engine().(*tree.BornEngine); !ok {
		t.Fatalf("expected BornEngine, got %T", auto.Engine())
	}

	fvals := make([]float64, auto.NFeatures())
	got := auto.PredictSingle(fvals, 0)
	want := bornCPU.PredictSingle(fvals, 0)
	if math.Abs(got-want) > 1e-5 {
		t.Errorf("predictions differ: auto=%v born=%v", got, want)
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
