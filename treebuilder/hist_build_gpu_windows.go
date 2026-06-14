//go:build windows && !js

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
	sumG, sumH, lambda float64,
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
			chunkFeats := feats[start:end]
			rows := make([]gpuHistF32Row, 0, len(chunkFeats))
			for _, feat := range chunkFeats {
				g32, h32, numBins, ok := accumulateOneHistWebGPU(gpu, feat, idx, gSub, hSub, n, cfg)
				if !ok {
					continue
				}
				rows = append(rows, gpuHistF32Row{
					feat: feat, histG: g32, histH: h32, numBins: numBins,
				})
				recordHistBuildWebGPU()
			}
			if len(rows) == 0 {
				continue
			}

			gainPicks := batchGainScanHistF32OnGPU(gpu, rows, sumG, sumH, lambda)
			gainHits := 0
			for _, row := range rows {
				if pick, ok := gainPicks[row.feat]; ok && pick.ok {
					out[row.feat] = gpuHistResult{
						splitIdx: pick.splitIdx,
						gain:     pick.gain,
						hasGain:  true,
						ok:       true,
					}
					gainHits++
					continue
				}
				out[row.feat] = gpuHistResult{
					histG: f32ToF64Slice(row.histG),
					histH: f32ToF64Slice(row.histH),
					ok:    true,
				}
			}
			if gainHits > 0 {
				recordGainScanWebGPUBatch(gainHits)
				setAccelWebGPUOK(true)
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
	sumG, sumH, lambda float64,
	cfg Config,
) (histG, histH []float64, ok bool) {
	_ = numBins
	batch := batchAccumulateHistWebGPU([]int{feat}, idx, grad, hess, sumG, sumH, lambda, cfg)
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
) (histG, histH []float32, numBins int, ok bool) {
	rowBins := cfg.GlobalBins.RowBin(feat)
	if rowBins == nil {
		return nil, nil, 0, false
	}
	var binsOK bool
	_, numBins, binsOK = cfg.GlobalBins.Lookup(feat)
	if !binsOK || numBins < bornHistMinBins {
		return nil, nil, 0, false
	}

	binIdx := gatherSubsetBins(rowBins, idx)
	destG := tensor.Zeros[float32](tensor.Shape{numBins}, gpu)
	destH := tensor.Zeros[float32](tensor.Shape{numBins}, gpu)
	idxT, err := tensor.FromSlice(binIdx, tensor.Shape{n}, gpu)
	if err != nil {
		return nil, nil, 0, false
	}
	srcG, err := tensor.FromSlice(gSub, tensor.Shape{n}, gpu)
	if err != nil {
		return nil, nil, 0, false
	}
	srcH, err := tensor.FromSlice(hSub, tensor.Shape{n}, gpu)
	if err != nil {
		return nil, nil, 0, false
	}

	outG := gpu.SelectAdd(destG.Raw(), 0, idxT.Raw(), srcG.Raw())
	outH := gpu.SelectAdd(destH.Raw(), 0, idxT.Raw(), srcH.Raw())
	return outG.AsFloat32(), outH.AsFloat32(), numBins, true
}
