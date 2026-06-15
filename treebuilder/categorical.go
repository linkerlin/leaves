package treebuilder

import (
	"github.com/linkerlin/leaves/data"
)

func bestCategoricalSplit(
	dm data.Matrix,
	idx []int,
	feat int,
	grad, hess []float64,
	sumG, sumH float64,
	row []float64,
	cfg Config,
) (gain float64, thr float64, left, right []int, ok bool) {
	uniq := uniqueIntValues(dm, idx, feat, row)
	if len(uniq) <= 1 {
		return 0, 0, nil, nil, false
	}
	bestGain := cfg.Gamma
	var bestLeft, bestRight []int
	for _, v := range uniq {
		if int(v) >= 32 {
			continue
		}
		l, r := splitCatEquals(dm, idx, feat, int(v), row)
		if len(l) == 0 || len(r) == 0 {
			continue
		}
		gl, hl := sumGradHess(l, grad, hess)
		gr, hr := sumGradHess(r, grad, hess)
		g := splitGain(gl, hl, gr, hr, sumG, sumH, cfg.Lambda)
		if g > bestGain {
			bestGain = g
			thr = v
			bestLeft = l
			bestRight = r
			ok = true
		}
	}
	return bestGain, thr, bestLeft, bestRight, ok
}

func uniqueIntValues(dm data.Matrix, idx []int, feat int, row []float64) []float64 {
	seen := make(map[int]struct{})
	var out []float64
	for _, i := range idx {
		_ = dm.Row(i, row)
		v := int(row[feat])
		if _, ok := seen[v]; ok {
			continue
		}
		seen[v] = struct{}{}
		out = append(out, float64(v))
	}
	return out
}

func splitCatEquals(dm data.Matrix, idx []int, feat int, val int, row []float64) (left, right []int) {
	for _, i := range idx {
		_ = dm.Row(i, row)
		if int(row[feat]) == val {
			left = append(left, i)
		} else {
			right = append(right, i)
		}
	}
	return left, right
}
