//go:build js

package treebuilder

import (
	"github.com/dmitryikh/leaves/data"
	"github.com/dmitryikh/leaves/tree"
)

// BornHistAvailable js 无 Born hist 加速。
func BornHistAvailable() bool { return false }

// WebGPUHistAvailable js 无 WebGPU hist。
func WebGPUHistAvailable() bool { return false }

func scanHistGains(histG, histH []float64, sumG, sumH, lambda float64, _ Config) (int, float64) {
	recordGainScanPureCPU()
	return scanHistGainsCPU(histG, histH, sumG, sumH, lambda)
}

// BuildHistGPU js 回退 CPU hist。
func BuildHistGPU(dm data.Matrix, indices []int, grad, hess []float64, cfg Config) *tree.TreeIR {
	cfg.UseGPUHist = false
	return BuildHist(dm, indices, grad, hess, cfg)
}
