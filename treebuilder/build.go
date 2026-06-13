package treebuilder

import (
	"github.com/dmitryikh/leaves/data"
	"github.com/dmitryikh/leaves/tree"
)

const (
	MethodExact   = "exact"
	MethodHist    = "hist"
	MethodAuto    = "auto"
	MethodGPUHist = "gpu_hist"
)

// ResolveMethod 解析建树算法；auto 时 n<5万用 exact，否则 hist。
func ResolveMethod(method string, nRow int) string {
	if method == MethodExact {
		return MethodExact
	}
	if method == MethodGPUHist {
		return MethodGPUHist
	}
	if method == MethodAuto {
		if nRow < 50000 {
			return MethodExact
		}
		return MethodHist
	}
	return MethodHist
}

// Build 按 method 选择建树算法。
func Build(dm data.Matrix, indices []int, grad, hess []float64, cfg Config, method string) *tree.TreeIR {
	switch ResolveMethod(method, dm.NumRow()) {
	case MethodExact:
		return BuildExact(dm, indices, grad, hess, cfg)
	case MethodGPUHist:
		if t := BuildHistGPU(dm, indices, grad, hess, cfg); t != nil {
			return t
		}
		fallthrough
	default:
		return BuildHist(dm, indices, grad, hess, cfg)
	}
}
