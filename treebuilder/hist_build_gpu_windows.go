//go:build windows

package treebuilder

import (
	"log"

	bornwebgpu "github.com/born-ml/born/backend/webgpu"
	"github.com/born-ml/born/tensor"
)

func batchAccumulateHistWebGPU(
	feats []int,
	idx []int,
	grad, hess []float64,
	cfg Config,
) map[int]gpuHistResult {
	if trainWebGPUDisabled || len(feats) == 0 {
		return nil
	}

	out := make(map[int]gpuHistResult)
	func() {
		defer func() {
			if r := recover(); r != nil {
				trainWebGPUDisabled = true
				log.Printf("[leaves/train] accel: webgpu hist batch error (%v), disabled for session", r)
				out = nil
			}
		}()

		trainWebGPUMu.Lock()
		defer trainWebGPUMu.Unlock()

		gpu, ready := initTrainWebGPU()
		if !ready || gpu == nil {
			out = nil
			return
		}

		n := len(idx)
		gSub := f64ToF32Slice(gatherSubsetF64(grad, idx))
		hSub := f64ToF32Slice(gatherSubsetF64(hess, idx))

		for start := 0; start < len(feats); start += gpuHistBatchMaxFeats {
			end := start + gpuHistBatchMaxFeats
			if end > len(feats) {
				end = len(feats)
			}
			for _, feat := range feats[start:end] {
				if r, ok := accumulateOneHistWebGPU(gpu, feat, idx, gSub, hSub, n, cfg); ok {
					out[feat] = r
					recordHistBuildWebGPU()
				}
			}
		}
		if len(out) > 0 {
			setAccelWebGPUOK(true)
		}
	}()
	return out
}

func accumulateHistWebGPU(
	feat int,
	idx []int,
	grad, hess []float64,
	numBins int,
	cfg Config,
) (histG, histH []float64, ok bool) {
	_ = numBins
	batch := batchAccumulateHistWebGPU([]int{feat}, idx, grad, hess, cfg)
	if batch == nil {
		return nil, nil, false
	}
	r, found := batch[feat]
	if !found || !r.ok {
		return nil, nil, false
	}
	return r.histG, r.histH, true
}

func accumulateOneHistWebGPU(
	gpu *bornwebgpu.Backend,
	feat int,
	idx []int,
	gSub, hSub []float32,
	n int,
	cfg Config,
) (gpuHistResult, bool) {
	rowBins := cfg.GlobalBins.RowBin(feat)
	if rowBins == nil {
		return gpuHistResult{}, false
	}
	_, numBins, ok := cfg.GlobalBins.Lookup(feat)
	if !ok || numBins < bornHistMinBins {
		return gpuHistResult{}, false
	}

	binIdx := gatherSubsetBins(rowBins, idx)
	destG := tensor.Zeros[float32](tensor.Shape{numBins}, gpu)
	destH := tensor.Zeros[float32](tensor.Shape{numBins}, gpu)
	idxT, err := tensor.FromSlice(binIdx, tensor.Shape{n}, gpu)
	if err != nil {
		return gpuHistResult{}, false
	}
	srcG, err := tensor.FromSlice(gSub, tensor.Shape{n}, gpu)
	if err != nil {
		return gpuHistResult{}, false
	}
	srcH, err := tensor.FromSlice(hSub, tensor.Shape{n}, gpu)
	if err != nil {
		return gpuHistResult{}, false
	}

	outG := gpu.SelectAdd(destG.Raw(), 0, idxT.Raw(), srcG.Raw())
	outH := gpu.SelectAdd(destH.Raw(), 0, idxT.Raw(), srcH.Raw())
	return gpuHistResult{
		histG: f32ToF64Slice(outG.AsFloat32()),
		histH: f32ToF64Slice(outH.AsFloat32()),
		ok:    true,
	}, true
}
