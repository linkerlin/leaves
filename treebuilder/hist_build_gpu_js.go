//go:build js

package treebuilder

func batchAccumulateHistWebGPU(
	feats []int,
	idx []int,
	grad, hess []float64,
	sumG, sumH, lambda float64,
	cfg Config,
) map[int]gpuHistResult {
	_ = feats
	_ = idx
	_ = grad
	_ = hess
	_ = sumG
	_ = sumH
	_ = lambda
	_ = cfg
	return nil
}

func accumulateHistWebGPU(
	feat int,
	idx []int,
	grad, hess []float64,
	numBins int,
	sumG, sumH, lambda float64,
	cfg Config,
) (histG, histH []float64, ok bool) {
	_ = feat
	_ = idx
	_ = grad
	_ = hess
	_ = numBins
	_ = sumG
	_ = sumH
	_ = lambda
	_ = cfg
	return nil, nil, false
}
