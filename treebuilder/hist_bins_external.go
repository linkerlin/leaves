package treebuilder

import (
	"github.com/linkerlin/leaves/data"
)

// BuildGlobalHistBinsExternal 多批次外存数据上构建全局直方图分箱。
func BuildGlobalHistBinsExternal(em data.ExternalMemoryMatrix, maxBin int, feats []int) *GlobalHistBins {
	if maxBin <= 0 {
		maxBin = 256
	}
	dm := data.Matrix(em)
	ncols := dm.NumCol()
	if len(feats) == 0 {
		feats = make([]int, ncols)
		for i := range feats {
			feats[i] = i
		}
	}
	out := &GlobalHistBins{
		byFeat: make([]FeatureBins, ncols),
		rowBin: make([][]int32, ncols),
	}
	n := dm.NumRow()
	row := make([]float64, ncols)
	for _, f := range feats {
		if f < 0 || f >= ncols {
			continue
		}
		vals := extractFeatureColumn(dm, f)
		cuts, numBins := histCutPoints(vals, maxBin)
		out.byFeat[f] = FeatureBins{Cuts: cuts, NumBins: numBins}
		if numBins <= 1 {
			continue
		}
		bins := make([]int32, n)
		for i := 0; i < n; i++ {
			_ = dm.Row(i, row)
			bins[i] = int32(valueToBinCuts(row[f], cuts))
		}
		out.rowBin[f] = bins
	}
	return out
}
