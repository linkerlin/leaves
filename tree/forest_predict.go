package tree

// ForestMargins 计算各 output group 的 raw margin（含 BaseScore）。
func ForestMargins(f *ForestIR, fvals []float64, nEstimators int) []float64 {
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
				addTreeToMargins(out, f, ti, fvals, 1.0)
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
			addTreeToMargins(out, f, treeIdx, fvals, coef)
		}
	}
	return out
}

// ForestMargin 单输出便捷封装。
func ForestMargin(f *ForestIR, fvals []float64, nEstimators int) float64 {
	m := ForestMargins(f, fvals, nEstimators)
	if len(m) == 0 {
		return 0
	}
	return m[0]
}

func classBaseScore(f *ForestIR, classIdx int) float64 {
	if len(f.BaseScores) > classIdx {
		return f.BaseScores[classIdx]
	}
	return f.BaseScore
}

func adjustNEstimators(f *ForestIR, nEstimators int) int {
	nEst := nEstimators
	if nEst <= 0 || nEst > f.NEstimators() {
		nEst = f.NEstimators()
	}
	return nEst
}

func weightDrop(f *ForestIR, treeIdx int) float64 {
	if treeIdx < len(f.WeightDrop) {
		return f.WeightDrop[treeIdx]
	}
	return 1.0
}

func forestTreeClassIndex(f *ForestIR, treeIdx int) int {
	if f == nil || f.NumOutputGroups <= 0 {
		return 0
	}
	if len(f.TreeInfo) > treeIdx {
		return f.TreeInfo[treeIdx]
	}
	return treeIdx % f.NumOutputGroups
}

func addTreeToMargins(out []float64, f *ForestIR, treeIdx int, fvals []float64, coef float64) {
	if treeIdx < 0 || treeIdx >= len(f.Trees) {
		return
	}
	t := &f.Trees[treeIdx]
	w := weightDrop(f, treeIdx) * coef
	if t.OutputDim > 1 {
		vec := treeVectorMargin(t, fvals)
		for d := 0; d < len(vec) && d < len(out); d++ {
			out[d] += vec[d] * w
		}
		return
	}
	k := forestTreeClassIndex(f, treeIdx)
	if k >= 0 && k < len(out) {
		out[k] += predictTreeScalar(t, fvals) * w
	}
}

func scaleExceptBase(f *ForestIR, out []float64, scale float64) {
	for k := range out {
		base := classBaseScore(f, k)
		out[k] = base + (out[k]-base)*scale
	}
}

// treeVectorMargin 返回向量叶各维 margin 贡献。
func treeVectorMargin(t *TreeIR, fvals []float64) []float64 {
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
	return treeLeafVector(t, walkTree(t, fvals))
}

// treeMargin 标量叶贡献（向量叶时返回 dim0，供旧式单输出路径）。
func treeMargin(t *TreeIR, fvals []float64) float64 {
	if t.OutputDim > 1 {
		vec := treeVectorMargin(t, fvals)
		if len(vec) > 0 {
			return vec[0]
		}
		return 0
	}
	return predictTreeScalar(t, fvals)
}

func predictTreeScalar(t *TreeIR, fvals []float64) float64 {
	if t.NumNodes == 0 {
		if len(t.LeafValue) > 0 {
			return t.LeafValue[0]
		}
		return 0
	}
	return treeLeafScalar(t, walkTree(t, fvals))
}

func walkTree(t *TreeIR, fvals []float64) int32 {
	nodeIdx := int32(0)
	for {
		leftIsLeaf := t.LeftChild[nodeIdx] < 0
		rightIsLeaf := t.RightChild[nodeIdx] < 0
		goLeft := treeDecision(t, int(nodeIdx), fvals)
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

func treeLeafScalar(t *TreeIR, leafNode int32) float64 {
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

// treeLeafVector 取向量叶；layout = [leafIdx * OutputDim + dim]。
func treeLeafVector(t *TreeIR, leafNode int32) []float64 {
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
