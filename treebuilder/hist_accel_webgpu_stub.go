//go:build !windows && !js

package treebuilder

func scanHistGainsWebGPU(histG, histH []float64, sumG, sumH, lambda float64) (int, float64, bool) {
	_ = histG
	_ = histH
	_ = sumG
	_ = sumH
	_ = lambda
	return -1, 0, false
}
