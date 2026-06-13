package tree

import "testing"

func TestSelectBackendLinear(t *testing.T) {
	caps := ModelCaps{IsLinear: true}
	got := SelectBackend(caps, DefaultWorkloadHint())
	if got != BackendNative {
		t.Errorf("linear: got %v want Native", got)
	}
}

func TestSelectBackendDefaultNumeric(t *testing.T) {
	forest := makeForest()
	caps := ModelCapsFromForest(forest, false, true)
	got := SelectBackend(caps, WorkloadHint{BatchSize: 8})
	if got != BackendNative {
		t.Errorf("small batch: got %v want Native", got)
	}
}

func TestSelectBackendWASM(t *testing.T) {
	forest := makeForest()
	caps := ModelCapsFromForest(forest, false, true)
	got := SelectBackend(caps, WorkloadHint{Target: DeployWASM})
	if got != BackendBornCPU {
		t.Errorf("WASM numeric: got %v want BornCPU", got)
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
	got := SelectBackend(caps, WorkloadHint{Target: DeployWASM})
	if got != BackendNative {
		t.Errorf("WASM catSmall: got %v want Native", got)
	}
}

func TestSelectBackendLargeBatchGPU(t *testing.T) {
	forest := makeForest()
	caps := ModelCapsFromForest(forest, false, true)
	got := SelectBackend(caps, WorkloadHint{BatchSize: 512, HasGPU: true})
	if got != BackendBornGPU {
		t.Errorf("large batch GPU: got %v want BornGPU", got)
	}
}

func TestSelectBackendLargeBatchNoGPU(t *testing.T) {
	forest := makeForest()
	caps := ModelCapsFromForest(forest, false, true)
	got := SelectBackend(caps, WorkloadHint{BatchSize: 512, HasGPU: false})
	if got != BackendNative {
		t.Errorf("large batch CPU: got %v want Native", got)
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
	if got != BackendBornCPU {
		t.Errorf("auto WASM: got %v want BornCPU", got)
	}
}
