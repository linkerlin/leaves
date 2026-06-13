package treebuilder

import (
	"github.com/dmitryikh/leaves/data"
)

const gpuHistBatchMaxFeats = 64

// gpuHistFullNodeMinRows 节点样本数达到此值时全部特征走 GPU；否则仅批处理前 gpuHistPartialMaxFeats 个。
const gpuHistFullNodeMinRows = 4096

const gpuHistPartialMaxFeats = 32

// gpuHistResult 预计算的 GPU/CPU 直方图。
type gpuHistResult struct {
	histG, histH []float64
	ok           bool
}

func gpuHistBatchEnabled(cfg Config) bool {
	if !cfg.UseGPUHist || effectiveAccelMode(cfg) != AccelModeWebGPU {
		return false
	}
	if histBinPolicy(cfg) != HistBinGlobal || cfg.GlobalBins == nil {
		return false
	}
	return true
}

func filterGPUHistFeats(feats []int, idx []int, depth int, cfg Config) []int {
	minN := gpuHistMinSamplesForDepth(depth)
	if len(idx) < minN {
		return nil
	}
	out := make([]int, 0, len(feats))
	for _, f := range feats {
		_, numBins, ok := cfg.GlobalBins.Lookup(f)
		if ok && numBins >= bornHistMinBins && cfg.GlobalBins.RowBin(f) != nil {
			out = append(out, f)
		}
	}
	if cap := gpuHistFeatCap(len(idx)); cap > 0 && len(out) > cap {
		out = out[:cap]
	}
	return out
}

func gpuHistFeatCap(lenIdx int) int {
	if lenIdx >= gpuHistFullNodeMinRows {
		return 0
	}
	return gpuHistPartialMaxFeats
}

func prebuildGPUHists(
	feats []int,
	idx []int,
	grad, hess []float64,
	depth int,
	cfg Config,
) map[int]gpuHistResult {
	if !gpuHistBatchEnabled(cfg) {
		return nil
	}
	gpuFeats := filterGPUHistFeats(feats, idx, depth, cfg)
	if len(gpuFeats) == 0 {
		return nil
	}
	return batchAccumulateHistWebGPU(gpuFeats, idx, grad, hess, cfg)
}

func histSplitFromFeat(
	dm data.Matrix,
	idx []int,
	feat int,
	grad, hess []float64,
	sumG, sumH float64,
	row []float64,
	cfg Config,
	prebuilt *gpuHistResult,
) histSplitPick {
	var cuts []float64
	var numBins int
	if histBinPolicy(cfg) == HistBinGlobal && cfg.GlobalBins != nil {
		var ok bool
		cuts, numBins, ok = cfg.GlobalBins.Lookup(feat)
		if !ok {
			return histSplitPick{}
		}
	} else {
		vals := make([]float64, 0, len(idx))
		for _, i := range idx {
			_ = dm.Row(i, row)
			vals = append(vals, row[feat])
		}
		cuts, numBins = histCutPoints(vals, cfg.MaxBin)
		if numBins <= 1 {
			return histSplitPick{}
		}
	}

	var histG, histH []float64
	if prebuilt != nil && prebuilt.ok {
		histG, histH = prebuilt.histG, prebuilt.histH
	} else {
		histG, histH = accumulateHistCPU(feat, idx, grad, hess, numBins, dm, row, cuts, cfg)
	}

	splitIdx, gain := scanHistGains(histG, histH, sumG, sumH, cfg.Lambda, cfg)
	if splitIdx < 0 || gain <= cfg.Gamma {
		return histSplitPick{}
	}
	left, right := splitIndices(dm, idx, feat, cuts[splitIdx], row)
	if len(left) == 0 || len(right) == 0 {
		return histSplitPick{}
	}
	return histSplitPick{
		feat:  feat,
		thr:   cuts[splitIdx],
		gain:  gain,
		left:  left,
		right: right,
		ok:    true,
	}
}
