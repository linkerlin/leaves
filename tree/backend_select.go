package tree

// DeployTarget 部署场景提示。
type DeployTarget int

const (
	DeployDefault DeployTarget = iota
	DeployWASM
)

// WorkloadHint 推理 workload 提示，供 BackendAuto 决策。
type WorkloadHint struct {
	BatchSize int
	HasGPU    bool
	Target    DeployTarget
}

func DefaultWorkloadHint() WorkloadHint {
	return WorkloadHint{BatchSize: 1, HasGPU: false, Target: DeployDefault}
}

// ModelCaps 后端选型所需的模型能力摘要。
type ModelCaps struct {
	IsLinear      bool
	IsNumericOnly bool
	Forest        *ForestIR
}

func ModelCapsFromForest(f *ForestIR, isLinear, isNumericOnly bool) ModelCaps {
	return ModelCaps{
		IsLinear:      isLinear,
		IsNumericOnly: isNumericOnly,
		Forest:        f,
	}
}

const autoBatchGPUThreshold = 256

func ResolveBackend(requested Backend, caps ModelCaps, hint WorkloadHint) Backend {
	if requested != BackendAuto {
		return requested
	}
	return SelectBackend(caps, hint)
}

// SelectBackend 按 v3.0 Born 迁移策略选择引擎。
func SelectBackend(caps ModelCaps, hint WorkloadHint) Backend {
	if caps.IsLinear || caps.Forest == nil {
		return BackendNative
	}

	if hint.Target == DeployWASM {
		if BornSupports(caps.Forest, BackendBornCPU) {
			return BackendBornCPU
		}
		return BackendNative
	}

	if hint.BatchSize >= autoBatchGPUThreshold && caps.IsNumericOnly && hint.HasGPU {
		if BornSupports(caps.Forest, BackendBornGPU) {
			return BackendBornGPU
		}
	}

	return BackendNative
}

// BornSupports 判断 Born 后端能否处理该森林。
func BornSupports(f *ForestIR, backend Backend) bool {
	if f == nil {
		return false
	}
	if backend == BackendBornCPU && forestHasCatSmall(f) {
		return false
	}
	if backend == BackendBornGPU {
		if forestHasCatSmall(f) || !BornWebGPUAvailable() {
			return false
		}
	}
	return true
}

func forestHasCatSmall(f *ForestIR) bool {
	for i := range f.Trees {
		t := &f.Trees[i]
		for j := range t.CatSmall {
			if t.CatSmall[j] {
				return true
			}
		}
	}
	return false
}
