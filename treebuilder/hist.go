package treebuilder

import (
	"sort"

	"github.com/dmitryikh/leaves/data"
	"github.com/dmitryikh/leaves/tree"
)

// BuildHist 直方图贪心建树（T4：默认多线程按特征块并行分裂搜索）。
func BuildHist(dm data.Matrix, indices []int, grad, hess []float64, cfg Config) *tree.TreeIR {
	if cfg.MaxDepth <= 0 {
		cfg.MaxDepth = 6
	}
	if cfg.MinHessian <= 0 {
		cfg.MinHessian = 1e-3
	}
	if cfg.Lambda < 0 {
		cfg.Lambda = 1.0
	}
	if cfg.LearningRate <= 0 {
		cfg.LearningRate = 0.3
	}
	if cfg.MaxBin <= 0 {
		cfg.MaxBin = 256
	}
	root := buildHistNode(dm, indices, grad, hess, 0, cfg)
	if root == nil {
		w := leafWeight(indices, grad, hess, cfg.Lambda) * cfg.LearningRate
		return tree.BuildTreeIR(nil, []float64{w}, nil, nil, 0)
	}
	nodes, leaves := flatten(root)
	return tree.BuildTreeIR(nodes, leaves, nil, nil, 0)
}

func buildHistNode(dm data.Matrix, idx []int, grad, hess []float64, depth int, cfg Config) *node {
	sumG, sumH := sumGradHess(idx, grad, hess)
	if sumH < cfg.MinHessian || depth >= cfg.MaxDepth || len(idx) <= 1 {
		return &node{
			leaf:    true,
			leafVal: leafWeightFromSums(sumG, sumH, cfg.Lambda) * cfg.LearningRate,
			sumHess: sumH,
		}
	}

	bestGain := cfg.Gamma
	var bestFeat int
	var bestThr float64
	var bestLeft, bestRight []int

	ncols := dm.NumCol()
	row := make([]float64, ncols)
	pick := findBestHistSplit(dm, idx, featureList(cfg, ncols), grad, hess, sumG, sumH, row, cfg)
	if pick.ok {
		bestGain = pick.gain
		bestFeat = pick.feat
		bestThr = pick.thr
		bestLeft = pick.left
		bestRight = pick.right
	}

	if bestGain <= cfg.Gamma {
		return &node{
			leaf:    true,
			leafVal: leafWeightFromSums(sumG, sumH, cfg.Lambda) * cfg.LearningRate,
			sumHess: sumH,
		}
	}

	return &node{
		feat:      bestFeat,
		threshold: bestThr,
		left:      buildHistNode(dm, bestLeft, grad, hess, depth+1, cfg),
		right:     buildHistNode(dm, bestRight, grad, hess, depth+1, cfg),
		sumHess:   sumH,
	}
}

func bestHistSplit(
	dm data.Matrix,
	idx []int,
	feat int,
	grad, hess []float64,
	sumG, sumH float64,
	row []float64,
	cfg Config,
) (bestFeat int, bestThr, bestGain float64, bestLeft, bestRight []int) {
	vals := make([]float64, 0, len(idx))
	for _, i := range idx {
		_ = dm.Row(i, row)
		vals = append(vals, row[feat])
	}
	cuts, numBins := histCutPoints(vals, cfg.MaxBin)
	if numBins <= 1 {
		return 0, 0, 0, nil, nil
	}

	histG := make([]float64, numBins)
	histH := make([]float64, numBins)

	for _, i := range idx {
		_ = dm.Row(i, row)
		b := valueToBinCuts(row[feat], cuts)
		histG[b] += grad[i]
		histH[b] += hess[i]
	}

	splitIdx, bestGain := scanHistGains(histG, histH, sumG, sumH, cfg.Lambda, cfg.UseGPUHist)
	if splitIdx < 0 || bestGain <= 0 {
		return 0, 0, 0, nil, nil
	}
	bestFeat = feat
	bestThr = cuts[splitIdx]
	bestLeft, bestRight = splitIndices(dm, idx, feat, bestThr, row)
	if len(bestLeft) == 0 || len(bestRight) == 0 {
		return 0, 0, 0, nil, nil
	}
	return bestFeat, bestThr, bestGain, bestLeft, bestRight
}

func histCutPoints(vals []float64, maxBin int) (cuts []float64, numBins int) {
	if len(vals) == 0 {
		return nil, 0
	}
	minV, maxV := vals[0], vals[0]
	for _, v := range vals[1:] {
		if v < minV {
			minV = v
		}
		if v > maxV {
			maxV = v
		}
	}
	if minV == maxV {
		return nil, 1
	}

	uniq := uniqueSorted(vals)
	numBins = len(uniq)
	if numBins > maxBin {
		numBins = maxBin
		width := (maxV - minV) / float64(numBins)
		cuts = make([]float64, numBins-1)
		for i := range cuts {
			cuts[i] = minV + width*float64(i+1)
		}
		return cuts, numBins
	}

	cuts = make([]float64, numBins-1)
	for i := 0; i < numBins-1; i++ {
		cuts[i] = (uniq[i] + uniq[i+1]) * 0.5
	}
	return cuts, numBins
}

func uniqueSorted(vals []float64) []float64 {
	cp := append([]float64(nil), vals...)
	sort.Float64s(cp)
	out := cp[:0]
	var prev float64
	for i, v := range cp {
		if i == 0 || v != prev {
			out = append(out, v)
			prev = v
		}
	}
	return out
}

func valueToBinCuts(v float64, cuts []float64) int {
	for i, c := range cuts {
		if v <= c {
			return i
		}
	}
	return len(cuts)
}
