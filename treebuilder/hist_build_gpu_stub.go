//go:build !windows

package treebuilder

func batchAccumulateHistWebGPU(
	feats []int,
	idx []int,
	grad, hess []float64,
	cfg Config,
) map[int]gpuHistResult {
	_ = feats
	_ = idx
	_ = grad
	_ = hess
	_ = cfg
	return nil
}

func accumulateHistWebGPU(
	feat int,
	idx []int,
	grad, hess []float64,
	numBins int,
	cfg Config,
) (histG, histH []float64, ok bool) {
	_ = feat
	_ = idx
	_ = grad
	_ = hess
	_ = numBins
	_ = cfg
	return nil, nil, false
}
