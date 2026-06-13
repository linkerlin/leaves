//go:build !born_train

package treebuilder

import (
	"github.com/dmitryikh/leaves/data"
	"github.com/dmitryikh/leaves/tree"
)

func scanHistGains(histG, histH []float64, sumG, sumH, lambda float64, useGPU bool) (int, float64) {
	_ = useGPU
	return scanHistGainsCPU(histG, histH, sumG, sumH, lambda)
}

func BornHistAvailable() bool { return false }

func BuildHistGPU(dm data.Matrix, indices []int, grad, hess []float64, cfg Config) *tree.TreeIR {
	_ = dm
	_ = indices
	_ = grad
	_ = hess
	_ = cfg
	return nil
}
