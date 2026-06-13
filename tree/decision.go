package tree

import "math"

// TreeDecision 导出节点路由（Born Native 回退路径使用）。
func TreeDecision(t *TreeIR, nodeIdx int, fvals []float64) bool {
	return treeDecision(t, nodeIdx, fvals)
}
func treeDecision(t *TreeIR, nodeIdx int, fvals []float64) bool {
	if nodeIdx < 0 || nodeIdx >= t.NumNodes {
		return false
	}
	feat := t.SplitFeature[nodeIdx]
	if int(feat) >= len(fvals) {
		return false
	}
	fval := fvals[feat]
	if nodeIdx < len(t.IsCategorical) && t.IsCategorical[nodeIdx] {
		return categoricalDecision(t, nodeIdx, fval)
	}
	return numericalDecision(t, nodeIdx, fval)
}

func numericalDecision(t *TreeIR, nodeIdx int, fval float64) bool {
	if math.IsNaN(fval) && (nodeIdx >= len(t.MissingNan) || !t.MissingNan[nodeIdx]) {
		fval = 0.0
	}
	missingZero := nodeIdx < len(t.MissingZero) && t.MissingZero[nodeIdx]
	missingNan := nodeIdx < len(t.MissingNan) && t.MissingNan[nodeIdx]
	if (missingZero && isZeroFval(fval)) || (missingNan && math.IsNaN(fval)) {
		if nodeIdx < len(t.DefaultLeft) {
			return t.DefaultLeft[nodeIdx]
		}
		return false
	}
	return fval <= t.SplitThreshold[nodeIdx]
}

func categoricalDecision(t *TreeIR, nodeIdx int, fval float64) bool {
	ifval := int32(fval)
	if ifval < 0 {
		return false
	}
	if math.IsNaN(fval) {
		if nodeIdx < len(t.MissingNan) && t.MissingNan[nodeIdx] {
			return false
		}
		ifval = 0
	}
	if nodeIdx < len(t.CatOneHot) && t.CatOneHot[nodeIdx] {
		return int32(t.SplitThreshold[nodeIdx]) == ifval
	}
	if nodeIdx < len(t.CatSmall) && t.CatSmall[nodeIdx] {
		return findInBitsetUint32(uint32(t.SplitThreshold[nodeIdx]), uint32(ifval))
	}
	catIdx := uint32(t.SplitThreshold[nodeIdx])
	return findInCatBitset(t, catIdx, uint32(ifval))
}

func findInCatBitset(t *TreeIR, catIdx uint32, pos uint32) bool {
	i1 := pos / 32
	idxS := t.CatBoundaries[catIdx]
	idxE := t.CatBoundaries[catIdx+1]
	if i1 >= (idxE - idxS) {
		return false
	}
	i2 := pos % 32
	return (t.CatThresholds[idxS+i1]>>i2)&1 > 0
}

func findInBitsetUint32(bits uint32, pos uint32) bool {
	if pos >= 32 {
		return false
	}
	return (bits>>pos)&1 > 0
}

func isZeroFval(fval float64) bool {
	const zeroThreshold = 1e-35
	return fval > -zeroThreshold && fval <= zeroThreshold
}
