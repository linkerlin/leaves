package treebuilder

import "github.com/linkerlin/leaves/data"

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
	k := cfg.OutputDim
	if k < 1 {
		k = 1
	}
	histG = make([]float64, numBins*k)
	histH = make([]float64, numBins*k)
	if cfg.GlobalBins != nil {
		if rowBins := cfg.GlobalBins.RowBin(feat); rowBins != nil {
			for _, i := range idx {
				bk := int(rowBins[i]) * k
				base := i * k
				for c := 0; c < k; c++ {
					histG[bk+c] += grad[base+c]
					histH[bk+c] += hess[base+c]
				}
			}
			recordHistBuildCPU()
			return histG, histH
		}
	}
	for _, i := range idx {
		_ = dm.Row(i, row)
		bk := valueToBinCuts(row[feat], cuts) * k
		base := i * k
		for c := 0; c < k; c++ {
			histG[bk+c] += grad[base+c]
			histH[bk+c] += hess[base+c]
		}
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
