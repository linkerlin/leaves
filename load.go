package leaves

import (
	"github.com/dmitryikh/leaves/io"
	"github.com/dmitryikh/leaves/model"
	"github.com/dmitryikh/leaves/tree"
)

// LoadOptions 模型加载选项（委托 io 包）。
type LoadOptions = io.LoadOptions

// Backend 推理后端。
type Backend = io.Backend

const (
	BackendNative  = io.BackendNative
	BackendBornCPU = io.BackendBornCPU
	BackendBornGPU = io.BackendBornGPU
	BackendAuto    = io.BackendAuto
)

// WorkloadHint 推理 workload 提示（BackendAuto 时使用）。
type WorkloadHint = tree.WorkloadHint

// LoadFromFile 自动检测格式并加载模型（v1.0 推荐入口）。
func LoadFromFile(filename string, opts *LoadOptions) (*model.Ensemble, error) {
	return io.LoadFromFile(filename, opts)
}

// DefaultLoadOptions 默认加载选项。
func DefaultLoadOptions() *LoadOptions {
	return io.DefaultLoadOptions()
}

// SelectBackend 根据 ModelIR 与 workload 选择后端。
func SelectBackend(ir *model.ModelIR, hint WorkloadHint) Backend {
	return io.SelectBackend(ir, hint)
}
