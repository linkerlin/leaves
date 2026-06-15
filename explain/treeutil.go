package explain

import (
	"math"

	"github.com/linkerlin/leaves/tree"
)

// treeDecision 与 NativeEngine 一致的节点路由（数值 + 分类 + 缺失值）。
func treeDecision(t *tree.TreeIR, nodeIdx int, fvals []float64) bool {
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

func numericalDecision(t *tree.TreeIR, nodeIdx int, fval float64) bool {
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

func categoricalDecision(t *tree.TreeIR, nodeIdx int, fval float64) bool {
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

func findInCatBitset(t *tree.TreeIR, catIdx uint32, pos uint32) bool {
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

func treeLeafValue(t *tree.TreeIR, nodeIdx int32) float64 {
	if nodeIdx < 0 {
		leafIdx := int(^nodeIdx)
		if leafIdx >= 0 && leafIdx < len(t.LeafValue) {
			return t.LeafValue[leafIdx]
		}
		return 0.0
	}
	if nodeIdx >= 0 && int(nodeIdx) < t.NumNodes {
		return 0.0
	}
	return 0.0
}

func predictTreeMargin(t *tree.TreeIR, fvals []float64) float64 {
	if t.NumNodes == 0 {
		if len(t.LeafValue) > 0 {
			return t.LeafValue[0]
		}
		return 0.0
	}
	nodeIdx := int32(0)
	for {
		leftIsLeaf := t.LeftChild[nodeIdx] < 0
		rightIsLeaf := t.RightChild[nodeIdx] < 0
		goLeft := treeDecision(t, int(nodeIdx), fvals)
		if goLeft {
			if leftIsLeaf {
				return treeLeafValue(t, t.LeftChild[nodeIdx])
			}
			nodeIdx = t.LeftChild[nodeIdx]
		} else {
			if rightIsLeaf {
				return treeLeafValue(t, t.RightChild[nodeIdx])
			}
			nodeIdx = t.RightChild[nodeIdx]
		}
	}
}

func predictForestMargin(f *tree.ForestIR, fvals []float64, nEstimators int) float64 {
	return tree.ForestMargin(f, fvals, nEstimators)
}

func treeClassIndex(f *tree.ForestIR, treeIdx int) int {
	if f == nil || f.NumOutputGroups <= 0 {
		return 0
	}
	if len(f.TreeInfo) > treeIdx {
		return f.TreeInfo[treeIdx]
	}
	return treeIdx % f.NumOutputGroups
}

func predictForestMarginClass(f *tree.ForestIR, fvals []float64, classIdx int, nEstimators int) float64 {
	m := tree.ForestMargins(f, fvals, nEstimators)
	if classIdx < 0 || classIdx >= len(m) {
		return 0
	}
	return m[classIdx]
}

func predictForestMargins(f *tree.ForestIR, fvals []float64, nEstimators int) []float64 {
	return tree.ForestMargins(f, fvals, nEstimators)
}
