package tree

import "fmt"

// DeployTarget 部署场景提示。
type DeployTarget int

const (
	DeployDefault DeployTarget = iota
	DeployWASM
)

// BackendAuto 2.0 阈值（历史文档兼容保留；2.1 起仅 small_batch/native_batch 分界）。
const (
	// AutoBatchCPUThreshold：batch < 此值 rule=small_batch；≥ 此值 rule=native_batch。
	// （2.0 曾以此为 BornCPU 选择线；2.1 实测未复现加速区，Born 转为显式/实测选型。）
	AutoBatchCPUThreshold = 64
	// AutoBatchGPUThreshold：Deprecated——2.1 起 Auto 不再据batch 选 BornGPU。
	AutoBatchGPUThreshold = 256
	// AutoSparseDensityMax：SparseDensity ∈ (0, 此值) 视为稀疏，走 Native。
	// SparseDensity=0 表示未知/稠密（不触发稀疏分支）。
	AutoSparseDensityMax = 0.15
)

// WorkloadHint 推理 workload 提示，供 BackendAuto 决策。
type WorkloadHint struct {
	BatchSize int
	HasGPU    bool
	Target    DeployTarget
	// SparseDensity 非零特征占比 ∈ (0,1]；0=未知（按稠密处理）。
	// 稀疏样本路径 Born 未优化，强制 Native。
	SparseDensity float64
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

// BackendDecision 是 SelectBackend 的可解释结果（文档 / Agent / 调试）。
type BackendDecision struct {
	Backend Backend
	Reason  string
	// Rule 稳定机器可读码（如 linear、small_batch、born_gpu）。
	Rule string
}

func ResolveBackend(requested Backend, caps ModelCaps, hint WorkloadHint) Backend {
	if requested != BackendAuto {
		return requested
	}
	return SelectBackend(caps, hint)
}

// SelectBackend 按 BackendAuto 2.0 决策表选择引擎（见 docs/backend-auto.md）。
func SelectBackend(caps ModelCaps, hint WorkloadHint) Backend {
	return SelectBackendExplained(caps, hint).Backend
}

// SelectBackendExplained 同 SelectBackend，并返回规则码与人类可读原因。
func SelectBackendExplained(caps ModelCaps, hint WorkloadHint) BackendDecision {
	if caps.IsLinear || caps.Forest == nil {
		return BackendDecision{
			Backend: BackendNative,
			Rule:    "linear_or_empty",
			Reason:  "线性模型或无森林 → Native（Born 仅树推理）",
		}
	}

	// —— WASM：永不选 GPU；数值树可 BornCPU ——
	if hint.Target == DeployWASM {
		if BornSupports(caps.Forest, BackendBornCPU) {
			return BackendDecision{
				Backend: BackendBornCPU,
				Rule:    "wasm_born_cpu",
				Reason:  "WASM 部署 + 数值树 → BornCPU",
			}
		}
		return BackendDecision{
			Backend: BackendNative,
			Rule:    "wasm_native_fallback",
			Reason:  "WASM 部署但森林含 cat-small 等 Born 不支持特性 → Native",
		}
	}

	// —— 稀疏：强制 Native ——
	if isSparseWorkload(hint) {
		return BackendDecision{
			Backend: BackendNative,
			Rule:    "sparse",
			Reason: fmt.Sprintf("SparseDensity=%.3f < %.2f → Native（稀疏路径未 Born 优化）",
				hint.SparseDensity, AutoSparseDensityMax),
		}
	}

	// —— 非数值 / cat-small：Born 不支持 ——
	if !caps.IsNumericOnly || !BornSupports(caps.Forest, BackendBornCPU) {
		return BackendDecision{
			Backend: BackendNative,
			Rule:    "non_numeric_or_unsupported",
			Reason:  "含类别分裂或 IsNumericOnly=false → Native",
		}
	}

	// —— 二轮 profiling（opt-in）：LEAVES_BACKEND_PROFILE=1 时以实测替代阈值 ——
	if backendProfilingEnabled() {
		return profiledDecision(caps, hint)
	}

	batch := hint.BatchSize
	if batch <= 0 {
		batch = 1
	}

	// —— 默认 Native（2.1 诚实化，2026-08-22 实测）——
	// 历史上 batch≥64 → BornCPU / batch≥256+GPU → BornGPU 的「加速区」
	// 在参考机上不可复现：lg_breast_cancer（39 树/30 特征/849 节点）与合成
	// 森林（100×63 节点）上，born v0.9.1 与 v0.9.23 的 BornCPU 在 batch
	// 64–4096 全段为 Native 的 0.03–0.16×（慢 6–30×）；BornGPU 计时异常
	// 且 batch≥256 挂起。详见 docs/benchmark-baseline.md §再测量。
	// 走 Born 的两条路：显式 BackendBornCPU/BornGPU，或
	// LEAVES_BACKEND_PROFILE=1（实测选型，测得更快才选）。
	if batch >= AutoBatchCPUThreshold {
		return BackendDecision{
			Backend: BackendNative,
			Rule:    "native_batch",
			Reason: fmt.Sprintf("batch=%d≥%d → Native（Born 加速区未实测复现；显式指定后端或设 LEAVES_BACKEND_PROFILE=1 实测选型）",
				batch, AutoBatchCPUThreshold),
		}
	}

	// —— 小 batch 在线 → Native ——
	return BackendDecision{
		Backend: BackendNative,
		Rule:    "small_batch",
		Reason: fmt.Sprintf("batch=%d<%d 在线小批 → Native golden",
			batch, AutoBatchCPUThreshold),
	}
}

func isSparseWorkload(hint WorkloadHint) bool {
	return hint.SparseDensity > 0 && hint.SparseDensity < AutoSparseDensityMax
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

// BackendName 返回稳定字符串（bench 记录 / JSON）。
func BackendName(b Backend) string {
	switch b {
	case BackendNative:
		return "native"
	case BackendBornCPU:
		return "born_cpu"
	case BackendBornGPU:
		return "born_gpu"
	case BackendAuto:
		return "auto"
	default:
		return fmt.Sprintf("backend_%d", int(b))
	}
}
