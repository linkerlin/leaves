package tree

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestSelectBackendLinear(t *testing.T) {
	caps := ModelCaps{IsLinear: true}
	d := SelectBackendExplained(caps, DefaultWorkloadHint())
	if d.Backend != BackendNative || d.Rule != "linear_or_empty" {
		t.Errorf("linear: got %+v", d)
	}
}

func TestSelectBackendDefaultNumeric(t *testing.T) {
	forest := makeForest()
	caps := ModelCapsFromForest(forest, false, true)
	d := SelectBackendExplained(caps, WorkloadHint{BatchSize: 8})
	if d.Backend != BackendNative || d.Rule != "small_batch" {
		t.Errorf("small batch: got %+v", d)
	}
}

func TestSelectBackendWASM(t *testing.T) {
	forest := makeForest()
	caps := ModelCapsFromForest(forest, false, true)
	for _, batch := range []int{0, 1, 63, 64, 512} {
		d := SelectBackendExplained(caps, WorkloadHint{Target: DeployWASM, BatchSize: batch})
		if d.Backend != BackendNative || d.Rule != "wasm_native" {
			t.Errorf("WASM batch=%d: got %+v want Native/wasm_native", batch, d)
		}
	}
}

func TestSelectBackendWASMCatSmall(t *testing.T) {
	nodes := []LgNodeData{
		{Feature: 0, Threshold: 3, Flags: flagCategorical | flagLeftLeaf | flagRightLeaf | flagCatSmall, Left: 0, Right: 1},
	}
	leafVals := []float64{1.0, -1.0}
	tir := BuildTreeIR(nodes, leafVals, nil, nil, 1)
	forest := &ForestIR{NumFeatures: 1, NumOutputGroups: 1, Trees: []TreeIR{*tir}}
	caps := ModelCapsFromForest(forest, false, false)
	d := SelectBackendExplained(caps, WorkloadHint{Target: DeployWASM})
	if d.Backend != BackendNative || d.Rule != "wasm_native" {
		t.Errorf("WASM catSmall: got %+v want Native/wasm_native", d)
	}
}

func TestSelectBackendLargeBatchGPU(t *testing.T) {
	forest := makeForest()
	caps := ModelCapsFromForest(forest, false, true)
	// 2.1 诚实化：GPU 大 batch 默认也 Native（BornGPU 计时异常实测；显式/profilng 路径不受影响）
	d := SelectBackendExplained(caps, WorkloadHint{BatchSize: 512, HasGPU: true})
	if d.Backend != BackendNative || d.Rule != "native_batch" {
		t.Errorf("large batch GPU: got %+v want native_batch", d)
	}
}

func TestSelectBackendLargeBatchNoGPU(t *testing.T) {
	forest := makeForest()
	caps := ModelCapsFromForest(forest, false, true)
	d := SelectBackendExplained(caps, WorkloadHint{BatchSize: 512, HasGPU: false})
	if d.Backend != BackendNative || d.Rule != "native_batch" {
		t.Errorf("large batch CPU: got %+v want native_batch", d)
	}
}

func TestSelectBackendMidBatchCPU(t *testing.T) {
	forest := makeForest()
	caps := ModelCapsFromForest(forest, false, true)
	d := SelectBackendExplained(caps, WorkloadHint{BatchSize: AutoBatchCPUThreshold})
	if d.Backend != BackendNative || d.Rule != "native_batch" {
		t.Errorf("mid batch: got %+v", d)
	}
	// 阈值下界 -1 → small_batch
	d = SelectBackendExplained(caps, WorkloadHint{BatchSize: AutoBatchCPUThreshold - 1})
	if d.Backend != BackendNative || d.Rule != "small_batch" {
		t.Errorf("below cpu threshold: got %+v", d)
	}
}

func TestSelectBackendSparse(t *testing.T) {
	forest := makeForest()
	caps := ModelCapsFromForest(forest, false, true)
	d := SelectBackendExplained(caps, WorkloadHint{
		BatchSize:     512,
		HasGPU:        true,
		SparseDensity: 0.05, // < AutoSparseDensityMax
	})
	if d.Backend != BackendNative || d.Rule != "sparse" {
		t.Errorf("sparse: got %+v want Native/sparse", d)
	}
	// 0 = 未知 → 不触发稀疏（native_batch 而非 sparse）
	d = SelectBackendExplained(caps, WorkloadHint{BatchSize: 128, SparseDensity: 0})
	if d.Rule != "native_batch" || d.Backend != BackendNative {
		t.Errorf("SparseDensity=0 should not force sparse rule: %+v", d)
	}
}

func TestSelectBackendNonNumeric(t *testing.T) {
	forest := makeForest()
	caps := ModelCapsFromForest(forest, false, false) // not numeric only
	d := SelectBackendExplained(caps, WorkloadHint{BatchSize: 512, HasGPU: true})
	if d.Backend != BackendNative || d.Rule != "non_numeric_or_unsupported" {
		t.Errorf("non-numeric: got %+v", d)
	}
}

func TestResolveBackendExplicit(t *testing.T) {
	forest := makeForest()
	caps := ModelCapsFromForest(forest, false, true)
	got := ResolveBackend(BackendBornCPU, caps, DefaultWorkloadHint())
	if got != BackendBornCPU {
		t.Errorf("explicit: got %v want BornCPU", got)
	}
}

func TestResolveBackendAuto(t *testing.T) {
	forest := makeForest()
	caps := ModelCapsFromForest(forest, false, true)
	got := ResolveBackend(BackendAuto, caps, WorkloadHint{Target: DeployWASM})
	if got != BackendNative {
		t.Errorf("auto WASM: got %v want Native", got)
	}
}

func TestBackendNameAndBenchRecord(t *testing.T) {
	if BackendName(BackendNative) != "native" || BackendName(BackendBornCPU) != "born_cpu" {
		t.Fatal(BackendName(BackendNative), BackendName(BackendBornCPU))
	}
	r := NewBenchRecord("predict/smoke/batch1", BackendNative, 1, 12345)
	r.AutoRule = "small_batch"
	r.Iters = 100
	b, err := r.MarshalJSONL()
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(b, &m); err != nil {
		t.Fatal(err)
	}
	if int(m["schema_version"].(float64)) != BenchRecordSchemaVersion {
		t.Fatalf("schema: %v", m["schema_version"])
	}
	if m["backend"] != "native" || m["auto_rule"] != "small_batch" {
		t.Fatalf("%s", b)
	}
	if !strings.Contains(string(b), "ns_per_op") {
		t.Fatal(string(b))
	}
}

// TestBackendAutoDecisionTable 锁定 2.1 决策表核心行（文档对照）。
func TestBackendAutoDecisionTable(t *testing.T) {
	forest := makeForest()
	num := ModelCapsFromForest(forest, false, true)
	table := []struct {
		hint WorkloadHint
		want Backend
		rule string
	}{
		{DefaultWorkloadHint(), BackendNative, "small_batch"},
		{WorkloadHint{BatchSize: 1}, BackendNative, "small_batch"},
		{WorkloadHint{BatchSize: 63}, BackendNative, "small_batch"},
		{WorkloadHint{BatchSize: 64}, BackendNative, "native_batch"},
		{WorkloadHint{BatchSize: 255, HasGPU: true}, BackendNative, "native_batch"},
		{WorkloadHint{BatchSize: 256, HasGPU: false}, BackendNative, "native_batch"},
		{WorkloadHint{BatchSize: 4096, HasGPU: true}, BackendNative, "native_batch"},
		{WorkloadHint{Target: DeployWASM, BatchSize: 1}, BackendNative, "wasm_native"},
		{WorkloadHint{Target: DeployWASM, BatchSize: 512}, BackendNative, "wasm_native"},
		{WorkloadHint{BatchSize: 1000, SparseDensity: 0.1}, BackendNative, "sparse"},
	}
	for _, row := range table {
		d := SelectBackendExplained(num, row.hint)
		if d.Backend != row.want || d.Rule != row.rule {
			t.Errorf("hint=%+v got backend=%v rule=%q want %v / %q (%s)",
				row.hint, d.Backend, d.Rule, row.want, row.rule, d.Reason)
		}
	}
}
