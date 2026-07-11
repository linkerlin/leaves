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
