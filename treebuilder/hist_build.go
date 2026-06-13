package treebuilder

import "github.com/dmitryikh/leaves/data"

const gpuHistMinSamples = 64 // 保留名供文档引用；运行时用 gpuHistMinSamplesForDepth

func gpuHistBuildEnabled(cfg Config) bool {
	return false // Phase 3：单特征 GPU hist 由 batchAccumulateHistWebGPU 统一处理
}

func accumulateHistCPU(
	feat int,
	idx []int,
	grad, hess []float64,
	numBins int,
	dm data.Matrix,
	row []float64,
	cuts []float64,
	cfg Config,
) (histG, histH []float64) {
	histG = make([]float64, numBins)
	histH = make([]float64, numBins)
	if cfg.GlobalBins != nil {
		if rowBins := cfg.GlobalBins.RowBin(feat); rowBins != nil {
			for _, i := range idx {
				b := rowBins[i]
				histG[b] += grad[i]
				histH[b] += hess[i]
			}
			recordHistBuildCPU()
			return histG, histH
		}
	}
	for _, i := range idx {
		_ = dm.Row(i, row)
		b := valueToBinCuts(row[feat], cuts)
		histG[b] += grad[i]
		histH[b] += hess[i]
	}
	recordHistBuildCPU()
	return histG, histH
}

func accumulateHist(
	feat int,
	idx []int,
	grad, hess []float64,
	numBins int,
	dm data.Matrix,
	row []float64,
	cuts []float64,
	cfg Config,
) (histG, histH []float64) {
	return accumulateHistCPU(feat, idx, grad, hess, numBins, dm, row, cuts, cfg)
}

func gatherSubsetF64(src []float64, idx []int) []float64 {
	out := make([]float64, len(idx))
	for j, i := range idx {
		out[j] = src[i]
	}
	return out
}

func gatherSubsetBins(rowBins []int32, idx []int) []int32 {
	out := make([]int32, len(idx))
	for j, i := range idx {
		out[j] = rowBins[i]
	}
	return out
}

func f64ToF32Slice(in []float64) []float32 {
	out := make([]float32, len(in))
	for i, v := range in {
		out[i] = float32(v)
	}
	return out
}

func f32ToF64Slice(in []float32) []float64 {
	out := make([]float64, len(in))
	for i, v := range in {
		out[i] = float64(v)
	}
	return out
}
