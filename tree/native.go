package tree

import (
	"fmt"
	"math"

	"github.com/dmitryikh/leaves/predict"
)

// 编译期检查：NativeEngine 实现 predict.Engine。
var _ predict.Engine = (*NativeEngine)(nil)

// NativeEngine 纯 Go 实现的树推理引擎，封装当前 leaves 的全部预测逻辑。
// 零外部依赖，作为 GoMLX 及其他后端的回退路径。
type NativeEngine struct {
	forest     *ForestIR
	transform  TransformFn
	outputType TransformType

	nRawOutputGroups int
	nOutputGroups    int
}

// NewNativeEngine 用原生 Go 推理逻辑创建引擎。
func NewNativeEngine(forest *ForestIR, transform TransformFn, outputType TransformType, nOutputGroups int) *NativeEngine {
	return &NativeEngine{
		forest:           forest,
		transform:        transform,
		outputType:       outputType,
		nRawOutputGroups: forest.NumOutputGroups,
		nOutputGroups:    nOutputGroups,
	}
}

// adjustNEstimators 将 nEstimators 钳制到有效范围。
func (e *NativeEngine) adjustNEstimators(nEstimators int) int {
	maxEstimators := e.forest.NEstimators()
	if nEstimators > 0 && nEstimators < maxEstimators {
		return nEstimators
	}
	return maxEstimators
}

// ---- Engine 接口实现 ----

func (e *NativeEngine) NOutputGroups() int    { return e.nOutputGroups }
func (e *NativeEngine) NRawOutputGroups() int  { return e.nRawOutputGroups }
func (e *NativeEngine) NFeatures() int         { return e.forest.NumFeatures }
func (e *NativeEngine) NEstimators() int       { return e.forest.NEstimators() }
func (e *NativeEngine) NLeaves() []int         { return e.forest.NLeaves() }
func (e *NativeEngine) Name() string           { return e.forest.Name }
func (e *NativeEngine) Close() error           { return nil }

// ---- 核心预测逻辑 ----

// Forest 返回底层 ForestIR（可解释性 API 使用）。
func (e *NativeEngine) Forest() *ForestIR { return e.forest }

// PredictSingle 单样本单输出预测。
func (e *NativeEngine) PredictSingle(fvals []float64, nEstimators int) float64 {
	if e.NOutputGroups() != 1 {
		return 0.0
	}
	if e.NFeatures() > len(fvals) {
		return 0.0
	}
	nEstimators = e.adjustNEstimators(nEstimators)

	if e.outputType == TransformLeafIndex {
		// leaf index 预测只支持多输出
		return 0.0
	}

	ret := [1]float64{0.0}
	e.predictInner(fvals, nEstimators, ret[:], 0)
	e.applyTransform(ret[:], ret[:], 0)
	return ret[0]
}

// Predict 单样本多输出预测。
func (e *NativeEngine) Predict(fvals []float64, nEstimators int, predictions []float64) error {
	if len(predictions) < e.NOutputGroups() {
		return fmt.Errorf("predictions slice too short (need at least %d)", e.NOutputGroups())
	}
	if e.NFeatures() > len(fvals) {
		return fmt.Errorf("incorrect number of features (%d)", len(fvals))
	}
	nEstimators = e.adjustNEstimators(nEstimators)

	if e.outputType == TransformLeafIndex {
		return e.predictLeafIndicesInner(fvals, nEstimators, predictions, 0)
	}

	e.predictInner(fvals, nEstimators, predictions, 0)
	e.applyTransform(predictions, predictions, 0)
	return nil
}

// PredictDense 稠密矩阵批量预测。
func (e *NativeEngine) PredictDense(
	vals []float64, nrows int, ncols int,
	predictions []float64,
	nEstimators int,
) error {
	if len(predictions) < e.NOutputGroups()*nrows {
		return fmt.Errorf("predictions slice too short (need at least %d)", e.NOutputGroups()*nrows)
	}
	if ncols == 0 || e.NFeatures() > ncols {
		return fmt.Errorf("incorrect number of columns")
	}
	nEstimators = e.adjustNEstimators(nEstimators)

	if e.outputType == TransformLeafIndex {
		for i := 0; i < nrows; i++ {
			fvals := vals[i*ncols : (i+1)*ncols]
			e.predictLeafIndicesInner(fvals, nEstimators, predictions, i*e.NOutputGroups())
		}
		return nil
	}

	for i := 0; i < nrows; i++ {
		fvals := vals[i*ncols : (i+1)*ncols]
		e.predictInner(fvals, nEstimators, predictions, i*e.NOutputGroups())
		e.applyTransform(predictions, predictions, i*e.NOutputGroups())
	}
	return nil
}

// PredictMarginDense 稠密矩阵 raw margin 预测（跳过 sigmoid/softmax）。
func (e *NativeEngine) PredictMarginDense(
	vals []float64, nrows int, ncols int,
	predictions []float64,
	nEstimators int,
) error {
	if len(predictions) < e.NRawOutputGroups()*nrows {
		return fmt.Errorf("predictions slice too short (need at least %d)", e.NRawOutputGroups()*nrows)
	}
	if ncols == 0 || e.NFeatures() > ncols {
		return fmt.Errorf("incorrect number of columns")
	}
	nEstimators = e.adjustNEstimators(nEstimators)
	g := e.NRawOutputGroups()
	for i := 0; i < nrows; i++ {
		fvals := vals[i*ncols : (i+1)*ncols]
		e.predictInner(fvals, nEstimators, predictions, i*g)
	}
	return nil
}

// PredictCSR CSR 稀疏矩阵批量预测。
func (e *NativeEngine) PredictCSR(
	indptr []int, cols []int, vals []float64,
	predictions []float64,
	nEstimators int,
) error {
	nRows := len(indptr) - 1
	if len(predictions) < e.NOutputGroups()*nRows {
		return fmt.Errorf("predictions slice too short (need at least %d)", e.NOutputGroups()*nRows)
	}
	nEstimators = e.adjustNEstimators(nEstimators)

	fvals := make([]float64, e.NFeatures())
	if e.outputType == TransformLeafIndex {
		for i := 0; i < nRows; i++ {
			e.resetFVals(fvals, true)
			start := indptr[i]
			end := indptr[i+1]
			for j := start; j < end; j++ {
				if cols[j] < len(fvals) {
					fvals[cols[j]] = vals[j]
				}
			}
			e.predictLeafIndicesInner(fvals, nEstimators, predictions, i*e.NOutputGroups())
		}
		return nil
	}

	for i := 0; i < nRows; i++ {
		e.resetFVals(fvals, true)
		start := indptr[i]
		end := indptr[i+1]
		for j := start; j < end; j++ {
			if cols[j] < len(fvals) {
				fvals[cols[j]] = vals[j]
			}
		}
		e.predictInner(fvals, nEstimators, predictions, i*e.NOutputGroups())
		e.applyTransform(predictions, predictions, i*e.NOutputGroups())
	}
	return nil
}

// PredictMarginCSR CSR 稀疏矩阵 raw margin 预测。
func (e *NativeEngine) PredictMarginCSR(
	indptr []int, cols []int, vals []float64,
	predictions []float64,
	nEstimators int,
) error {
	nRows := len(indptr) - 1
	if len(predictions) < e.NRawOutputGroups()*nRows {
		return fmt.Errorf("predictions slice too short (need at least %d)", e.NRawOutputGroups()*nRows)
	}
	nEstimators = e.adjustNEstimators(nEstimators)
	g := e.NRawOutputGroups()
	fvals := make([]float64, e.NFeatures())
	for i := 0; i < nRows; i++ {
		e.resetFVals(fvals, true)
		start := indptr[i]
		end := indptr[i+1]
		for j := start; j < end; j++ {
			if cols[j] < len(fvals) {
				fvals[cols[j]] = vals[j]
			}
		}
		e.predictInner(fvals, nEstimators, predictions, i*g)
	}
	return nil
}

// PredictLeafIndicesDense 稠密矩阵叶子索引预测。
func (e *NativeEngine) PredictLeafIndicesDense(
	vals []float64, nrows int, ncols int,
	predictions []float64,
) error {
	return e.PredictDense(vals, nrows, ncols, predictions, e.NEstimators())
}

// PredictLeafIndicesCSR CSR 稀疏矩阵叶子索引预测。
func (e *NativeEngine) PredictLeafIndicesCSR(
	indptr []int, cols []int, vals []float64,
	predictions []float64,
) error {
	return e.PredictCSR(indptr, cols, vals, predictions, e.NEstimators())
}

// ---- 内部实现 ----

// predictInner 计算单样本原始预测值（不应用变换）。
func (e *NativeEngine) predictInner(fvals []float64, nEstimators int, predictions []float64, startIndex int) {
	margins := ForestMargins(e.forest, fvals, nEstimators)
	for k, v := range margins {
		predictions[startIndex+k] = v
	}
}

// predictLeafIndicesInner 计算叶子索引（不应用变换）。
func (e *NativeEngine) predictLeafIndicesInner(fvals []float64, nEstimators int, predictions []float64, startIndex int) error {
	f := e.forest
	nResults := f.NumOutputGroups * nEstimators
	for k := 0; k < nResults; k++ {
		predictions[startIndex+k] = 0.0
	}

	for i := 0; i < nEstimators; i++ {
		for k := 0; k < f.NumOutputGroups; k++ {
			treeIdx := i*f.NumOutputGroups + k
			leafIdx := e.predictTreeIndex(&f.Trees[treeIdx], fvals)
			predictions[startIndex+k*nEstimators+i] = float64(leafIdx)
		}
	}
	return nil
}

// applyTransform 将变换函数应用于原始预测值。
func (e *NativeEngine) applyTransform(rawPredictions []float64, output []float64, startIndex int) {
	if e.transform == nil || e.outputType == TransformRaw {
		return
	}
	raw := rawPredictions[startIndex : startIndex+e.nRawOutputGroups]
	e.transform(raw, output, startIndex)
}

// resetFVals 重置特征值数组。useNaN=true 用于 XGBoost（NaN 表示缺失）。
func (e *NativeEngine) resetFVals(fvals []float64, useNaN bool) {
	if useNaN {
		for j := 0; j < len(fvals); j++ {
			fvals[j] = math.NaN()
		}
	} else {
		for j := 0; j < len(fvals); j++ {
			fvals[j] = 0.0
		}
	}
}

// ---- 单棵树推理 ----

// predictTree 在单棵树上推理，返回叶子值。
func (e *NativeEngine) predictTree(t *TreeIR, fvals []float64) float64 {
	if t.NumNodes == 0 {
		// 单节点树（常数树）
		if len(t.LeafValue) > 0 {
			return t.LeafValue[0]
		}
		return 0.0
	}

	nodeIdx := int32(0)
	for {
		// 检查是否是叶子（子索引为负表示叶子，负值的补码为叶子值表索引）
		// TreeIR 使用 -1 表示左子为叶，LeftChild 存的是叶子值数组中的索引
		leftIsLeaf := t.LeftChild[nodeIdx] < 0
		rightIsLeaf := t.RightChild[nodeIdx] < 0

		goLeft := e.treeDecision(t, int(nodeIdx), fvals)

		if goLeft {
			if leftIsLeaf {
				leafIdx := int(^t.LeftChild[nodeIdx]) // 取反得到正索引
				if leafIdx < len(t.LeafValue) {
					return t.LeafValue[leafIdx]
				}
				return 0.0
			}
			nodeIdx = t.LeftChild[nodeIdx]
		} else {
			if rightIsLeaf {
				leafIdx := int(^t.RightChild[nodeIdx])
				if leafIdx < len(t.LeafValue) {
					return t.LeafValue[leafIdx]
				}
				return 0.0
			}
			nodeIdx = t.RightChild[nodeIdx]
		}
	}
}

// predictTreeIndex 在单棵树上推理，返回叶子索引。
func (e *NativeEngine) predictTreeIndex(t *TreeIR, fvals []float64) int32 {
	if t.NumNodes == 0 {
		return 0
	}

	nodeIdx := int32(0)
	for {
		leftIsLeaf := t.LeftChild[nodeIdx] < 0
		rightIsLeaf := t.RightChild[nodeIdx] < 0

		goLeft := e.treeDecision(t, int(nodeIdx), fvals)

		if goLeft {
			if leftIsLeaf {
				return int32(^t.LeftChild[nodeIdx])
			}
			nodeIdx = t.LeftChild[nodeIdx]
		} else {
			if rightIsLeaf {
				return int32(^t.RightChild[nodeIdx])
			}
			nodeIdx = t.RightChild[nodeIdx]
		}
	}
}

// treeDecision 决定在给定节点走左分支还是右分支。
func (e *NativeEngine) treeDecision(t *TreeIR, nodeIdx int, fvals []float64) bool {
	return treeDecision(t, nodeIdx, fvals)
}
