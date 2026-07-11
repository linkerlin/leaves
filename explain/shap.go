package explain

import (
	"fmt"

	"github.com/linkerlin/leaves/tree"
)

const maxTreeDepth = 64

// TreeExplainer 基于 Tree SHAP 的可解释器（CPU，margin 空间）。
type TreeExplainer struct {
	forest      *tree.ForestIR
	nFeatures   int
	nEstimators int
	// LIB-22 缓存：按树预计算节点覆盖权重与背景（全零）margin。
	nodeWCache  [][]float64
	treeBase    []float64
	pathBufMax  int
	cacheReady  bool
}

// NewTreeExplainer 创建 Tree SHAP 解释器。
func NewTreeExplainer(f *tree.ForestIR) *TreeExplainer {
	if f == nil {
		return nil
	}
	n := f.NumFeatures
	if n <= 0 {
		n = maxFeatureIndex(f) + 1
	}
	e := &TreeExplainer{
		forest:      f,
		nFeatures:   n,
		nEstimators: f.NEstimators(),
	}
	e.ensureCache()
	return e
}

// ensureCache 预计算每棵树的节点权重与背景 margin（一次；多样本 SHAP 复用）。
func (e *TreeExplainer) ensureCache() {
	if e == nil || e.forest == nil || e.cacheReady {
		return
	}
	nt := len(e.forest.Trees)
	e.nodeWCache = make([][]float64, nt)
	e.treeBase = make([]float64, nt)
	bg := make([]float64, e.nFeatures)
	maxPath := 0
	for ti := range e.forest.Trees {
		t := &e.forest.Trees[ti]
		e.nodeWCache[ti] = computeNodeWeightsArr(t)
		e.treeBase[ti] = predictTreeMargin(t, bg)
		if n := pathBufLen(t); n > maxPath {
			maxPath = n
		}
	}
	e.pathBufMax = maxPath
	if e.pathBufMax < 16 {
		e.pathBufMax = 16
	}
	e.cacheReady = true
}

// ExpectedValue 返回背景（全零特征）上的 margin 预测（含 BaseScore）。
func (e *TreeExplainer) ExpectedValue() float64 {
	vals := e.ExpectedValues()
	if len(vals) == 0 {
		return 0
	}
	return vals[0]
}

// ExpectedValues 返回每个 output group 的背景 margin（全零特征）。
func (e *TreeExplainer) ExpectedValues() []float64 {
	if e == nil || e.forest == nil {
		return nil
	}
	bg := make([]float64, e.nFeatures)
	return predictForestMargins(e.forest, bg, 0)
}

// ShapleyValuesMulticlass 计算多类 Tree SHAP，返回 [sample][class][feature]（margin 空间）。
// 注：返回形状为 [sample][class][feature]（与注释历史一致以测试为准见 shap_multiclass）。
func (e *TreeExplainer) ShapleyValuesMulticlass(features [][]float64) ([][][]float64, error) {
	if e == nil || e.forest == nil {
		return nil, fmt.Errorf("nil explainer")
	}
	e.ensureCache()
	g := e.forest.NumOutputGroups
	if g <= 0 {
		g = 1
	}
	out := make([][][]float64, len(features))
	treePhi := make([]float64, e.nFeatures)
	pathBuf := make([]pathElement, e.pathBufMax)
	xMissing := make([]bool, e.nFeatures)
	for i, x := range features {
		if len(x) < e.nFeatures {
			return nil, fmt.Errorf("sample %d: need %d features, got %d", i, e.nFeatures, len(x))
		}
		for j := 0; j < e.nFeatures; j++ {
			xMissing[j] = x[j] != x[j] // NaN
		}
		phi := make([][]float64, g)
		for k := 0; k < g; k++ {
			phi[k] = make([]float64, e.nFeatures)
		}
		for ti := range e.forest.Trees {
			classIdx := treeClassIndex(e.forest, ti)
			clearFloats(treePhi)
			t := &e.forest.Trees[ti]
			treeShapFastReuse(t, x, xMissing, e.nodeWCache[ti], pathBuf, treePhi)
			treeMargin := predictTreeMargin(t, x)
			treeBase := e.treeBase[ti]
			treeSum := 0.0
			for _, v := range treePhi {
				treeSum += v
			}
			// residual 归到根分裂特征（path 上首特征恒为根）
			residual := (treeMargin - treeBase) - treeSum
			if t.NumNodes > 0 {
				f := int(t.SplitFeature[0])
				if f >= 0 && f < len(treePhi) {
					treePhi[f] += residual
				}
			}
			for f, v := range treePhi {
				phi[classIdx][f] += v
			}
		}
		out[i] = phi
	}
	return out, nil
}

// ShapleyValues 计算精确 Tree SHAP（单输出，margin 空间，tree_path_dependent + 背景残差校正）。
func (e *TreeExplainer) ShapleyValues(features [][]float64) ([][]float64, error) {
	if e == nil || e.forest == nil {
		return nil, fmt.Errorf("nil explainer")
	}
	if e.forest.NumOutputGroups > 1 {
		return nil, fmt.Errorf("multiclass: use ShapleyValuesMulticlass")
	}
	e.ensureCache()
	out := make([][]float64, len(features))
	treePhi := make([]float64, e.nFeatures)
	pathBuf := make([]pathElement, e.pathBufMax)
	xMissing := make([]bool, e.nFeatures)
	for i, x := range features {
		if len(x) < e.nFeatures {
			return nil, fmt.Errorf("sample %d: need %d features, got %d", i, e.nFeatures, len(x))
		}
		for j := 0; j < e.nFeatures; j++ {
			xMissing[j] = x[j] != x[j]
		}
		phi := make([]float64, e.nFeatures)
		for ti := range e.forest.Trees {
			clearFloats(treePhi)
			t := &e.forest.Trees[ti]
			treeShapFastReuse(t, x, xMissing, e.nodeWCache[ti], pathBuf, treePhi)
			treeMargin := predictTreeMargin(t, x)
			treeBase := e.treeBase[ti]
			treeSum := 0.0
			for _, v := range treePhi {
				treeSum += v
			}
			residual := (treeMargin - treeBase) - treeSum
			if t.NumNodes > 0 {
				f := int(t.SplitFeature[0])
				if f >= 0 && f < len(treePhi) {
					treePhi[f] += residual
				}
			}
			for f, v := range treePhi {
				phi[f] += v
			}
		}
		out[i] = phi
	}
	return out, nil
}

// InteractionValues 计算 Tree SHAP 交互值（单输出，margin，tree_path_dependent，背景=全零）。
// 返回 [sample][feature][feature]；对角元满足 Σ_j φ_ij = 主效应 φ_i。
func (e *TreeExplainer) InteractionValues(features [][]float64) ([][][]float64, error) {
	if e == nil || e.forest == nil {
		return nil, fmt.Errorf("nil explainer")
	}
	if e.forest.NumOutputGroups > 1 {
		return nil, fmt.Errorf("multiclass: use InteractionValuesMulticlass")
	}
	main, err := e.ShapleyValues(features)
	if err != nil {
		return nil, err
	}
	out := make([][][]float64, len(features))
	for i, x := range features {
		n := e.nFeatures
		mat := make([][]float64, n)
		for j := range mat {
			mat[j] = make([]float64, n)
		}
		for ti := range e.forest.Trees {
			treeInteractionFast(&e.forest.Trees[ti], x, mat)
		}
		// 对角元 = 主效应 − 非对角列和，保证 Σ_j φ_ij = φ_i
		for fi := 0; fi < n; fi++ {
			rowSum := 0.0
			for fj := 0; fj < n; fj++ {
				if fi != fj {
					rowSum += mat[fi][fj]
				}
			}
			mat[fi][fi] = main[i][fi] - rowSum
		}
		out[i] = mat
	}
	return out, nil
}

// ApproximateContributions Saabas 近似贡献（沿实例路径，margin 空间）。
func (e *TreeExplainer) ApproximateContributions(features [][]float64) ([][]float64, error) {
	if e == nil || e.forest == nil {
		return nil, fmt.Errorf("nil explainer")
	}
	if e.forest.NumOutputGroups > 1 {
		return nil, fmt.Errorf("multiclass: use ApproximateContributionsMulticlass")
	}
	out := make([][]float64, len(features))
	for i, x := range features {
		if len(x) < e.nFeatures {
			return nil, fmt.Errorf("sample %d: need %d features, got %d", i, e.nFeatures, len(x))
		}
		phi := make([]float64, e.nFeatures)
		for ti := range e.forest.Trees {
			saabasTree(&e.forest.Trees[ti], x, phi)
		}
		out[i] = phi
	}
	return out, nil
}

// InteractionValuesMulticlass 多类交互 SHAP，返回 [sample][class][feature][feature]。
func (e *TreeExplainer) InteractionValuesMulticlass(features [][]float64) ([][][][]float64, error) {
	main, err := e.ShapleyValuesMulticlass(features)
	if err != nil {
		return nil, err
	}
	g := e.forest.NumOutputGroups
	if g <= 0 {
		g = 1
	}
	out := make([][][][]float64, len(features))
	for i, x := range features {
		perClass := make([][][]float64, g)
		for k := 0; k < g; k++ {
			n := e.nFeatures
			mat := make([][]float64, n)
			for j := range mat {
				mat[j] = make([]float64, n)
			}
			for ti := range e.forest.Trees {
				if treeClassIndex(e.forest, ti) != k {
					continue
				}
				treeInteractionFast(&e.forest.Trees[ti], x, mat)
			}
			for fi := 0; fi < n; fi++ {
				rowSum := 0.0
				for fj := 0; fj < n; fj++ {
					if fi != fj {
						rowSum += mat[fi][fj]
					}
				}
				mat[fi][fi] = main[i][k][fi] - rowSum
			}
			perClass[k] = mat
		}
		out[i] = perClass
	}
	return out, nil
}

// ApproximateContributionsMulticlass Saabas 多类近似，返回 [sample][class][feature]。
func (e *TreeExplainer) ApproximateContributionsMulticlass(features [][]float64) ([][][]float64, error) {
	if e == nil || e.forest == nil {
		return nil, fmt.Errorf("nil explainer")
	}
	g := e.forest.NumOutputGroups
	if g <= 0 {
		g = 1
	}
	out := make([][][]float64, len(features))
	for i, x := range features {
		if len(x) < e.nFeatures {
			return nil, fmt.Errorf("sample %d: need %d features, got %d", i, e.nFeatures, len(x))
		}
		phi := make([][]float64, g)
		for k := 0; k < g; k++ {
			phi[k] = make([]float64, e.nFeatures)
		}
		for ti := range e.forest.Trees {
			classIdx := treeClassIndex(e.forest, ti)
			saabasTree(&e.forest.Trees[ti], x, phi[classIdx])
		}
		out[i] = phi
	}
	return out, nil
}

// treeInteractionInterventional 对单棵树累加路径特征对的交互 SHAP（非对角，对称）。
func treeInteractionInterventional(t *tree.TreeIR, x []float64, phi [][]float64) {
	path := treePathFeatures(t, x)
	m := len(path)
	if m < 2 {
		return
	}
	for pi := 0; pi < m; pi++ {
		for pj := pi + 1; pj < m; pj++ {
			fi := path[pi]
			fj := path[pj]
			if fi < 0 || fj < 0 || fi >= len(phi) || fj >= len(phi[fi]) {
				continue
			}
			var sum float64
			for mask := 0; mask < (1 << m); mask++ {
				if (mask>>pi)&1 != 0 || (mask>>pj)&1 != 0 {
					continue
				}
				coalitionSize := 0
				for k := 0; k < m; k++ {
					if (mask>>k)&1 != 0 {
						coalitionSize++
					}
				}
				w := interactionWeight(m, coalitionSize)
				mask11 := mask | (1 << pi) | (1 << pj)
				mask10 := mask | (1 << pi)
				mask01 := mask | (1 << pj)
				v11 := predictTreeMargin(t, maskFeatures(x, path, mask11, len(x)))
				v10 := predictTreeMargin(t, maskFeatures(x, path, mask10, len(x)))
				v01 := predictTreeMargin(t, maskFeatures(x, path, mask01, len(x)))
				v00 := predictTreeMargin(t, maskFeatures(x, path, mask, len(x)))
				sum += w * (v11 - v10 - v01 + v00)
			}
			phi[fi][fj] += sum
			phi[fj][fi] += sum
		}
	}
}

func interactionWeight(pathLen, coalitionSize int) float64 {
	if pathLen < 2 || coalitionSize < 0 || coalitionSize > pathLen-2 {
		return 0
	}
	return factorial(coalitionSize) * factorial(pathLen-coalitionSize-2) / (2 * factorial(pathLen-1))
}

func factorial(n int) float64 {
	r := 1.0
	for k := 2; k <= n; k++ {
		r *= float64(k)
	}
	return r
}

// treeShapInterventional 对单棵树做路径精确 SHAP。
func treeShapInterventional(t *tree.TreeIR, x []float64, phi []float64) {
	path := treePathFeatures(t, x)
	if len(path) == 0 {
		return
	}
	m := len(path)
	for i := 0; i < m; i++ {
		feat := path[i]
		var sum float64
		for mask := 0; mask < (1 << m); mask++ {
			if (mask>>i)&1 == 0 {
				continue
			}
			coalitionSize := 0
			for j := 0; j < m; j++ {
				if (mask>>j)&1 != 0 {
					coalitionSize++
				}
			}
			weight := shapleyWeight(m, coalitionSize)
			xMasked := maskFeatures(x, path, mask, len(x))
			vWith := predictTreeMargin(t, xMasked)
			maskWithout := mask & ^(1 << i)
			xWithout := maskFeatures(x, path, maskWithout, len(x))
			vWithout := predictTreeMargin(t, xWithout)
			sum += weight * (vWith - vWithout)
		}
		if feat >= 0 && feat < len(phi) {
			phi[feat] += sum
		}
	}
}

func shapleyWeight(pathLen, coalitionSize int) float64 {
	if coalitionSize <= 0 || coalitionSize > pathLen {
		return 0
	}
	return factorial(coalitionSize-1) * factorial(pathLen-coalitionSize) / factorial(pathLen)
}

func maskFeatures(x []float64, path []int, mask int, n int) []float64 {
	out := make([]float64, n) // interventional 背景：未激活特征置 0
	for j, feat := range path {
		if (mask>>j)&1 != 0 && feat >= 0 && feat < n {
			out[feat] = x[feat]
		}
	}
	return out
}

func treePathFeatures(t *tree.TreeIR, x []float64) []int {
	if t.NumNodes == 0 {
		return nil
	}
	var path []int
	nodeIdx := int32(0)
	for {
		feat := int(t.SplitFeature[nodeIdx])
		path = append(path, feat)
		goLeft := treeDecision(t, int(nodeIdx), x)
		leftIsLeaf := t.LeftChild[nodeIdx] < 0
		rightIsLeaf := t.RightChild[nodeIdx] < 0
		if goLeft {
			if leftIsLeaf {
				break
			}
			nodeIdx = t.LeftChild[nodeIdx]
		} else {
			if rightIsLeaf {
				break
			}
			nodeIdx = t.RightChild[nodeIdx]
		}
	}
	return path
}

// saabasTree 沿实例路径的 Saabas 近似。
func saabasTree(t *tree.TreeIR, x []float64, phi []float64) {
	path := treePathFeatures(t, x)
	if len(path) == 0 {
		return
	}
	leafVal := predictTreeMargin(t, x)
	share := leafVal / float64(len(path))
	for _, feat := range path {
		if feat >= 0 && feat < len(phi) {
			phi[feat] += share
		}
	}
}
