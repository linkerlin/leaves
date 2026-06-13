package quantize

import (
	"math"

	"github.com/dmitryikh/leaves/tree"
)

// forestMarginsQ 与 tree.ForestMargins 等价，但数值分裂使用量化阈值。
func forestMarginsQ(qf *QuantizedForest, fvals []float64, nEstimators int) []float64 {
	f := &qf.Forest
	if f == nil {
		return nil
	}
	g := f.NumOutputGroups
	if g <= 0 {
		g = 1
	}
	out := make([]float64, g)
	for k := 0; k < g; k++ {
		out[k] = classBaseScore(f, k)
	}
	nEst := adjustNEstimators(f, nEstimators)
	if nEst <= 0 {
		return out
	}
	if len(f.IterationIndptr) > 1 {
		for iter := 0; iter < nEst; iter++ {
			if iter+1 >= len(f.IterationIndptr) {
				break
			}
			for ti := f.IterationIndptr[iter]; ti < f.IterationIndptr[iter+1]; ti++ {
				addTreeToMarginsQ(out, qf, ti, fvals, 1.0)
			}
		}
		if f.AverageOutput {
			scaleExceptBase(f, out, 1.0/float64(nEst))
		}
		return out
	}
	coef := 1.0
	if f.AverageOutput {
		coef = 1.0 / float64(nEst)
	}
	for i := 0; i < nEst; i++ {
		for k := 0; k < g; k++ {
			treeIdx := i*g + k
			if treeIdx >= len(f.Trees) {
				continue
			}
			addTreeToMarginsQ(out, qf, treeIdx, fvals, coef)
		}
	}
	return out
}

func addTreeToMarginsQ(out []float64, qf *QuantizedForest, treeIdx int, fvals []float64, coef float64) {
	f := &qf.Forest
	if treeIdx < 0 || treeIdx >= len(f.Trees) {
		return
	}
	t := &f.Trees[treeIdx]
	w := weightDrop(f, treeIdx) * coef
	if t.OutputDim > 1 {
		vec := treeVectorMarginQ(qf, treeIdx, t, fvals)
		for d := 0; d < len(vec) && d < len(out); d++ {
			out[d] += vec[d] * w
		}
		return
	}
	k := forestTreeClassIndex(f, treeIdx)
	if k >= 0 && k < len(out) {
		out[k] += predictTreeScalarQ(qf, treeIdx, t, fvals) * w
	}
}

func predictTreeScalarQ(qf *QuantizedForest, treeIdx int, t *tree.TreeIR, fvals []float64) float64 {
	if t.NumNodes == 0 {
		if len(t.LeafValue) > 0 {
			return t.LeafValue[0]
		}
		return 0
	}
	return treeLeafScalarQ(t, walkTreeQ(qf, treeIdx, t, fvals))
}

func treeVectorMarginQ(qf *QuantizedForest, treeIdx int, t *tree.TreeIR, fvals []float64) []float64 {
	if t == nil {
		return nil
	}
	dim := t.OutputDim
	if dim <= 0 {
		dim = 1
	}
	if t.NumNodes == 0 {
		out := make([]float64, dim)
		for d := 0; d < dim && d < len(t.LeafValue); d++ {
			out[d] = t.LeafValue[d]
		}
		return out
	}
	return treeLeafVectorQ(t, walkTreeQ(qf, treeIdx, t, fvals))
}

func walkTreeQ(qf *QuantizedForest, treeIdx int, t *tree.TreeIR, fvals []float64) int32 {
	nodeIdx := int32(0)
	for {
		leftIsLeaf := t.LeftChild[nodeIdx] < 0
		rightIsLeaf := t.RightChild[nodeIdx] < 0
		goLeft := treeDecisionQ(qf, treeIdx, t, int(nodeIdx), fvals)
		if goLeft {
			if leftIsLeaf {
				return t.LeftChild[nodeIdx]
			}
			nodeIdx = t.LeftChild[nodeIdx]
		} else {
			if rightIsLeaf {
				return t.RightChild[nodeIdx]
			}
			nodeIdx = t.RightChild[nodeIdx]
		}
	}
}

func treeDecisionQ(qf *QuantizedForest, treeIdx int, t *tree.TreeIR, nodeIdx int, fvals []float64) bool {
	if nodeIdx < 0 || nodeIdx >= t.NumNodes {
		return false
	}
	feat := t.SplitFeature[nodeIdx]
	if int(feat) >= len(fvals) {
		return false
	}
	fval := fvals[feat]
	if isCategoricalNode(t, nodeIdx) {
		return categoricalDecisionQ(t, nodeIdx, fval)
	}
	return numericalDecisionQ(qf, treeIdx, t, nodeIdx, fval)
}

func numericalDecisionQ(qf *QuantizedForest, treeIdx int, t *tree.TreeIR, nodeIdx int, fval float64) bool {
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
	th := t.SplitThreshold[nodeIdx]
	if treeIdx < len(qf.Quantized) && nodeIdx < len(qf.Quantized[treeIdx]) && qf.Quantized[treeIdx][nodeIdx] {
		feat := int(t.SplitFeature[nodeIdx])
		if feat >= 0 && feat < len(qf.FeatureMin) {
			th = decodeThreshold(qf.QThreshold[treeIdx][nodeIdx], qf.FeatureMin[feat], qf.FeatureSpan[feat], qf.levels)
		}
	}
	return fval <= th
}

func categoricalDecisionQ(t *tree.TreeIR, nodeIdx int, fval float64) bool {
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

func treeLeafScalarQ(t *tree.TreeIR, leafNode int32) float64 {
	if leafNode >= 0 {
		return 0
	}
	leafIdx := int(^leafNode)
	if leafIdx < 0 || leafIdx >= len(t.LeafValue) {
		return 0
	}
	if t.OutputDim > 1 {
		return t.LeafValue[leafIdx*t.OutputDim]
	}
	return t.LeafValue[leafIdx]
}

func treeLeafVectorQ(t *tree.TreeIR, leafNode int32) []float64 {
	dim := t.OutputDim
	if dim <= 0 {
		dim = 1
	}
	out := make([]float64, dim)
	if leafNode >= 0 {
		return out
	}
	leafIdx := int(^leafNode)
	base := leafIdx * dim
	for d := 0; d < dim; d++ {
		if base+d < len(t.LeafValue) {
			out[d] = t.LeafValue[base+d]
		}
	}
	return out
}

func classBaseScore(f *tree.ForestIR, classIdx int) float64 {
	if len(f.BaseScores) > classIdx {
		return f.BaseScores[classIdx]
	}
	return f.BaseScore
}

func adjustNEstimators(f *tree.ForestIR, nEstimators int) int {
	maxEstimators := f.NEstimators()
	if nEstimators > 0 && nEstimators < maxEstimators {
		return nEstimators
	}
	return maxEstimators
}

func weightDrop(f *tree.ForestIR, treeIdx int) float64 {
	if treeIdx < len(f.WeightDrop) {
		return f.WeightDrop[treeIdx]
	}
	return 1.0
}

func forestTreeClassIndex(f *tree.ForestIR, treeIdx int) int {
	if len(f.TreeInfo) > treeIdx {
		return f.TreeInfo[treeIdx]
	}
	if f.NumOutputGroups <= 0 {
		return 0
	}
	return treeIdx % f.NumOutputGroups
}

func scaleExceptBase(f *tree.ForestIR, out []float64, scale float64) {
	for k := range out {
		base := classBaseScore(f, k)
		out[k] = base + (out[k]-base)*scale
	}
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
