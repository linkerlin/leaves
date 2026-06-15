package explain

import "github.com/linkerlin/leaves/tree"

// treeInteractionFast 用条件 Tree SHAP 计算交互值（O(T·D²·U)，U=树内唯一特征数）。
// 仅累加非对角 φ[i][j]（i≠j），对角由 InteractionValues 从主效应校正。
func treeInteractionFast(t *tree.TreeIR, x []float64, phi [][]float64) {
	if t == nil || t.NumNodes == 0 || len(phi) == 0 {
		return
	}
	n := len(phi)
	uniq := uniqueTreeFeatures(t)
	if len(uniq) == 0 {
		return
	}

	maxd := t.MaxDepth + 2
	if maxd < 4 {
		maxd = 4
	}
	pathBuf := make([]pathElement, (maxd*(maxd+1))/2)
	nodeW := computeNodeWeights(t)
	xMissing := xMissingMask(x)

	phiOn := make([]float64, n)
	phiOff := make([]float64, n)
	for _, j := range uniq {
		if j < 0 || j >= n {
			continue
		}
		clearFloats(phiOn)
		clearFloats(phiOff)
		treeShapConditioned(t, x, xMissing, nodeW, pathBuf, phiOn, 1, j)
		treeShapConditioned(t, x, xMissing, nodeW, pathBuf, phiOff, -1, j)
		for l := 0; l < n; l++ {
			if l == j {
				continue
			}
			val := (phiOn[l] - phiOff[l]) * 0.5
			if j < len(phi) && l < len(phi[j]) {
				phi[j][l] += val
			}
			if l < len(phi) && j < len(phi[l]) {
				phi[l][j] += val
			}
		}
	}
}

func treeShapConditioned(t *tree.TreeIR, x []float64, xMissing []bool, nodeW map[int32]float64, pathBuf []pathElement, phi []float64, condition, conditionFeature int) {
	treeShapRecursive(t, x, xMissing, nodeW, phi, 0, 0, pathBuf, 1, 1, -1, condition, conditionFeature, 1)
}

func uniqueTreeFeatures(t *tree.TreeIR) []int {
	seen := make(map[int]bool)
	var out []int
	for i := 0; i < t.NumNodes; i++ {
		f := int(t.SplitFeature[i])
		if f < 0 || seen[f] {
			continue
		}
		seen[f] = true
		out = append(out, f)
	}
	return out
}

func xMissingMask(x []float64) []bool {
	m := make([]bool, len(x))
	for i, v := range x {
		m[i] = isMissingFval(v)
	}
	return m
}

func isMissingFval(v float64) bool {
	return v != v // NaN
}

func clearFloats(a []float64) {
	for i := range a {
		a[i] = 0
	}
}
