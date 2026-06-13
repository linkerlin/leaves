package explain

import (
	"math"

	"github.com/dmitryikh/leaves/tree"
)

// pathElement Tree SHAP 唯一路径元素（Lundberg et al. 2018）。
type pathElement struct {
	featureIndex int
	zeroFraction float64
	oneFraction  float64
	pweight      float64
}

// treeShapFast 使用 O(T·D²) Tree SHAP（tree_path_dependent，SumHess 作覆盖权重）。
func treeShapFast(t *tree.TreeIR, x []float64, phi []float64) {
	if t == nil || t.NumNodes == 0 {
		return
	}
	maxd := t.MaxDepth + 2
	if maxd < 4 {
		maxd = 4
	}
	pathBuf := make([]pathElement, (maxd*(maxd+1))/2)
	nodeW := computeNodeWeights(t)
	xMissing := make([]bool, len(x))
	for i, v := range x {
		xMissing[i] = math.IsNaN(v)
	}
	treeShapRecursive(t, x, xMissing, nodeW, phi, 0, 0, pathBuf, 1, 1, -1, 0, 0, 1)
}

func computeNodeWeights(t *tree.TreeIR) map[int32]float64 {
	w := make(map[int32]float64)
	var weight func(node int32) float64
	weight = func(node int32) float64 {
		if v, ok := w[node]; ok {
			return v
		}
		if node < 0 {
			w[node] = 1.0
			return 1.0
		}
		ni := int(node)
		if ni < len(t.SumHess) && t.SumHess[ni] > 0 {
			w[node] = t.SumHess[ni]
			return w[node]
		}
		lw := weight(t.LeftChild[ni])
		rw := weight(t.RightChild[ni])
		w[node] = lw + rw
		if w[node] <= 0 {
			w[node] = 1.0
		}
		return w[node]
	}
	weight(0)
	return w
}

func extendPath(path []pathElement, uniqueDepth int, zeroFraction, oneFraction float64, featureIndex int) {
	path[uniqueDepth].featureIndex = featureIndex
	path[uniqueDepth].zeroFraction = zeroFraction
	path[uniqueDepth].oneFraction = oneFraction
	path[uniqueDepth].pweight = 0
	if uniqueDepth == 0 {
		path[0].pweight = 1
		return
	}
	path[uniqueDepth].pweight = 0
	for i := uniqueDepth - 1; i >= 0; i-- {
		path[i+1].pweight += oneFraction * path[i].pweight * float64(i+1) / float64(uniqueDepth+1)
		path[i].pweight = zeroFraction * path[i].pweight * float64(uniqueDepth-i) / float64(uniqueDepth+1)
	}
}

func unwindPath(path []pathElement, uniqueDepth, pathIndex int) {
	oneFraction := path[pathIndex].oneFraction
	zeroFraction := path[pathIndex].zeroFraction
	nextOnePortion := path[uniqueDepth].pweight
	for i := uniqueDepth - 1; i >= 0; i-- {
		if oneFraction != 0 {
			tmp := path[i].pweight
			path[i].pweight = nextOnePortion * float64(uniqueDepth+1) / (float64(i+1) * oneFraction)
			nextOnePortion = tmp - path[i].pweight*zeroFraction*float64(uniqueDepth-i)/float64(uniqueDepth+1)
		} else if zeroFraction != 0 {
			path[i].pweight = path[i].pweight * float64(uniqueDepth+1) / (zeroFraction * float64(uniqueDepth-i))
		}
	}
	for i := pathIndex; i < uniqueDepth; i++ {
		path[i] = path[i+1]
	}
}

func unwoundPathSum(path []pathElement, uniqueDepth, pathIndex int) float64 {
	oneFraction := path[pathIndex].oneFraction
	zeroFraction := path[pathIndex].zeroFraction
	nextOnePortion := path[uniqueDepth].pweight
	total := 0.0
	if oneFraction != 0 {
		for i := uniqueDepth - 1; i >= 0; i-- {
			tmp := nextOnePortion / (float64(i+1) * oneFraction)
			total += tmp
			nextOnePortion = path[i].pweight - tmp*zeroFraction*float64(uniqueDepth-i)
		}
	} else if zeroFraction != 0 {
		for i := uniqueDepth - 1; i >= 0; i-- {
			total += path[i].pweight / (zeroFraction * float64(uniqueDepth-i))
		}
	}
	return total * float64(uniqueDepth+1)
}

func treeShapRecursive(
	t *tree.TreeIR, x []float64, xMissing []bool, nodeW map[int32]float64,
	phi []float64,
	nodeIndex int32, uniqueDepth int,
	parentPath []pathElement,
	parentZeroFraction, parentOneFraction float64,
	parentFeatureIndex int,
	condition int, conditionFeature int,
	conditionFraction float64,
) {
	if conditionFraction == 0 {
		return
	}
	uniquePath := parentPath[uniqueDepth+1:]
	copy(uniquePath[:uniqueDepth+1], parentPath[:uniqueDepth+1])

	if condition == 0 || parentFeatureIndex != conditionFeature {
		extendPath(uniquePath, uniqueDepth, parentZeroFraction, parentOneFraction, parentFeatureIndex)
	}

	if nodeIndex < 0 {
		leafVal := treeLeafValue(t, nodeIndex)
		for i := 1; i <= uniqueDepth; i++ {
			w := unwoundPathSum(uniquePath, uniqueDepth, i)
			el := uniquePath[i]
			if el.featureIndex < 0 || el.featureIndex >= len(phi) {
				continue
			}
			scale := w * (el.oneFraction - el.zeroFraction) * conditionFraction
			phi[el.featureIndex] += scale * leafVal
		}
		return
	}

	ni := int(nodeIndex)
	if ni >= t.NumNodes {
		return
	}
	splitFeat := int(t.SplitFeature[ni])
	hot, cold := hotColdChildren(t, ni, x, xMissing)

	w := nodeW[nodeIndex]
	if w <= 0 {
		w = 1
	}
	hotW := nodeW[hot]
	coldW := nodeW[cold]
	if hotW <= 0 {
		hotW = 1
	}
	if coldW <= 0 {
		coldW = 1
	}
	hotZeroFraction := hotW / w
	coldZeroFraction := coldW / w
	incomingZeroFraction := 1.0
	incomingOneFraction := 1.0

	pathIndex := 0
	for ; pathIndex <= uniqueDepth; pathIndex++ {
		if uniquePath[pathIndex].featureIndex == splitFeat {
			break
		}
	}
	if pathIndex <= uniqueDepth {
		incomingZeroFraction = uniquePath[pathIndex].zeroFraction
		incomingOneFraction = uniquePath[pathIndex].oneFraction
		unwindPath(uniquePath, uniqueDepth, pathIndex)
		uniqueDepth--
	}

	hotCond := conditionFraction
	coldCond := conditionFraction
	if condition > 0 && splitFeat == conditionFeature {
		coldCond = 0
		uniqueDepth--
	} else if condition < 0 && splitFeat == conditionFeature {
		hotCond *= hotZeroFraction
		coldCond *= coldZeroFraction
		uniqueDepth--
	}

	treeShapRecursive(t, x, xMissing, nodeW, phi, hot, uniqueDepth+1, uniquePath,
		hotZeroFraction*incomingZeroFraction, incomingOneFraction, splitFeat,
		condition, conditionFeature, hotCond)
	treeShapRecursive(t, x, xMissing, nodeW, phi, cold, uniqueDepth+1, uniquePath,
		coldZeroFraction*incomingZeroFraction, 0, splitFeat,
		condition, conditionFeature, coldCond)
}

func hotColdChildren(t *tree.TreeIR, ni int, x []float64, xMissing []bool) (hot, cold int32) {
	left := t.LeftChild[ni]
	right := t.RightChild[ni]
	feat := int(t.SplitFeature[ni])
	if feat < len(xMissing) && xMissing[feat] {
		if ni < len(t.DefaultLeft) && t.DefaultLeft[ni] {
			return left, right
		}
		return right, left
	}
	if treeDecision(t, ni, x) {
		return left, right
	}
	return right, left
}
