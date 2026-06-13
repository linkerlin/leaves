package treebuilder

import (
	"github.com/dmitryikh/leaves/data"
)

const (
	HistBinGlobal  = "global"
	HistBinPerNode = "per_node"
)

// FeatureBins 单列特征的全局直方图分箱。
type FeatureBins struct {
	Cuts    []float64
	NumBins int
}

// GlobalHistBins 训练期全局分箱缓存（每特征一次 histCutPoints + 行级 bin 索引）。
type GlobalHistBins struct {
	byFeat []FeatureBins
	rowBin [][]int32 // rowBin[feat][row]
}

func histBinPolicy(cfg Config) string {
	if cfg.HistBinPolicy != "" {
		return cfg.HistBinPolicy
	}
	return HistBinGlobal
}

// BuildGlobalHistBins 对 feats 中每列在全量数据上计算切点；feats 为空则覆盖全部列。
func BuildGlobalHistBins(dm data.Matrix, maxBin int, feats []int) *GlobalHistBins {
	if maxBin <= 0 {
		maxBin = 256
	}
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
	for _, f := range feats {
		if f < 0 || f >= ncols {
			continue
		}
		vals := extractFeatureColumn(dm, f)
		cuts, numBins := histCutPoints(vals, maxBin)
		out.byFeat[f] = FeatureBins{Cuts: cuts, NumBins: numBins}
		if numBins > 1 {
			out.rowBin[f] = computeRowBins(vals, cuts)
		}
	}
	return out
}

// RowBin 返回特征 f 的预计算行级 bin 索引。
func (g *GlobalHistBins) RowBin(feat int) []int32 {
	if g == nil || feat < 0 || feat >= len(g.rowBin) {
		return nil
	}
	return g.rowBin[feat]
}

func computeRowBins(vals []float64, cuts []float64) []int32 {
	bins := make([]int32, len(vals))
	for i, v := range vals {
		bins[i] = int32(valueToBinCuts(v, cuts))
	}
	return bins
}

// Lookup 返回特征 f 的全局切点；常量列返回 ok=false。
func (g *GlobalHistBins) Lookup(feat int) (cuts []float64, numBins int, ok bool) {
	if g == nil || feat < 0 || feat >= len(g.byFeat) {
		return nil, 0, false
	}
	fb := g.byFeat[feat]
	if fb.NumBins <= 1 {
		return nil, 0, false
	}
	return fb.Cuts, fb.NumBins, true
}

func ensureGlobalBins(dm data.Matrix, cfg *Config) {
	if histBinPolicy(*cfg) != HistBinGlobal {
		return
	}
	if cfg.GlobalBins != nil {
		return
	}
	cfg.GlobalBins = BuildGlobalHistBins(dm, cfg.MaxBin, featureList(*cfg, dm.NumCol()))
}

func extractFeatureColumn(dm data.Matrix, feat int) []float64 {
	if d, ok := dm.(*data.Dense); ok {
		n := d.Rows
		vals := make([]float64, n)
		for i := 0; i < n; i++ {
			vals[i] = d.Data[i*d.Cols+feat]
		}
		return vals
	}
	n := dm.NumRow()
	ncol := dm.NumCol()
	vals := make([]float64, n)
	row := make([]float64, ncol)
	for i := 0; i < n; i++ {
		_ = dm.Row(i, row)
		vals[i] = row[feat]
	}
	return vals
}
