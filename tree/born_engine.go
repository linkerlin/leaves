//go:build !js

// Package tree — Born 张量推理引擎（ForestIR 直读；CPU / WebGPU）。
package tree

import (
	"fmt"
	"math"

	borncpu "github.com/born-ml/born/backend/cpu"
)

var _ Engine = (*BornEngine)(nil)

// BornEngine 基于 Born 张量后端的树推理引擎，直接持有 ForestIR。
type BornEngine struct {
	forest           *ForestIR
	transform        TransformFn
	outputType       TransformType
	nRawOutputGroups int
	nOutputGroups    int
	cpu              *borncpu.Backend
	gpu              any // *webgpu.Backend on Windows when UseGPU 成功
	usingGPU         bool
}

// BornConfig Born 引擎配置。
type BornConfig struct {
	UseGPU bool
}

// NewBornEngine 创建 Born 推理引擎。
func NewBornEngine(forest *ForestIR, transform TransformFn, outputType TransformType, nOutputGroups int, cfg *BornConfig) (*BornEngine, error) {
	if forest == nil {
		return nil, fmt.Errorf("born: nil forest")
	}
	useGPU := cfg != nil && cfg.UseGPU
	e := &BornEngine{
		forest:           forest,
		transform:        transform,
		outputType:       outputType,
		nRawOutputGroups: forest.NumOutputGroups,
		nOutputGroups:    nOutputGroups,
	}
	if useGPU {
		gpu, err := bornOpenWebGPU()
		if err == nil && gpu != nil {
			e.gpu = gpu
			e.usingGPU = true
			return e, nil
		}
	}
	e.cpu = borncpu.New()
	return e, nil
}

// BornUsingGPU 引擎是否实际使用 WebGPU 张量后端。
func (e *BornEngine) BornUsingGPU() bool {
	return e != nil && e.usingGPU
}

func (e *BornEngine) NOutputGroups() int    { return e.nOutputGroups }
func (e *BornEngine) NRawOutputGroups() int { return e.nRawOutputGroups }
func (e *BornEngine) NFeatures() int        { return e.forest.NumFeatures }
func (e *BornEngine) NEstimators() int      { return e.forest.NEstimators() }
func (e *BornEngine) NLeaves() []int        { return e.forest.NLeaves() }
func (e *BornEngine) Name() string          { return e.forest.Name }
func (e *BornEngine) Forest() *ForestIR     { return e.forest }
func (e *BornEngine) Close() error {
	if e != nil {
		bornCloseWebGPU(e.gpu)
		e.gpu = nil
	}
	return nil
}

func (e *BornEngine) adjNEst(n int) int {
	return adjustNEstimators(e.forest, n)
}

func (e *BornEngine) PredictSingle(fvals []float64, nEstimators int) float64 {
	if e.NOutputGroups() != 1 {
		return 0
	}
	if e.NFeatures() > len(fvals) {
		return 0
	}
	preds := make([]float64, 1)
	_ = e.Predict(fvals, nEstimators, preds)
	return preds[0]
}

func (e *BornEngine) Predict(fvals []float64, nEstimators int, predictions []float64) error {
	if len(predictions) < e.NOutputGroups() {
		return fmt.Errorf("predictions slice too short")
	}
	if e.NFeatures() > len(fvals) {
		return fmt.Errorf("incorrect number of features")
	}
	nEst := e.adjNEst(nEstimators)
	if e.outputType == TransformLeafIndex {
		return e.predictLeafIndicesRow(fvals, nEst, predictions, 0)
	}
	m, err := e.bornMargins(fvals, 1, len(fvals), nEst)
	if err != nil || len(m) == 0 {
		return err
	}
	copy(predictions, m[0])
	e.applyTransform(predictions, 0)
	return nil
}

func (e *BornEngine) PredictDense(vals []float64, nrows, ncols int, predictions []float64, nEstimators int) error {
	if len(predictions) < e.NOutputGroups()*nrows {
		return fmt.Errorf("predictions slice too short")
	}
	if ncols == 0 || e.NFeatures() > ncols {
		return fmt.Errorf("incorrect number of columns")
	}
	nEst := e.adjNEst(nEstimators)
	if e.outputType == TransformLeafIndex {
		for i := 0; i < nrows; i++ {
			fv := vals[i*ncols : (i+1)*ncols]
			if err := e.predictLeafIndicesRow(fv, nEst, predictions, i*e.NOutputGroups()); err != nil {
				return err
			}
		}
		return nil
	}
	margins, err := e.bornMargins(vals, nrows, ncols, nEst)
	if err != nil {
		return err
	}
	g := e.NOutputGroups()
	for i := 0; i < nrows; i++ {
		copy(predictions[i*g:(i+1)*g], margins[i])
		e.applyTransform(predictions, i*g)
	}
	return nil
}

func (e *BornEngine) PredictCSR(indptr, cols []int, vals, predictions []float64, nEstimators int) error {
	nRows := len(indptr) - 1
	nf := e.NFeatures()
	dense := make([]float64, nRows*nf)
	for i := 0; i < nRows; i++ {
		row := dense[i*nf : (i+1)*nf]
		for j := range row {
			row[j] = math.NaN()
		}
		start := indptr[i]
		end := indptr[i+1]
		for j := start; j < end; j++ {
			if cols[j] < nf {
				row[cols[j]] = vals[j]
			}
		}
	}
	return e.PredictDense(dense, nRows, nf, predictions, nEstimators)
}

func (e *BornEngine) PredictLeafIndicesDense(vals []float64, nrows, ncols int, predictions []float64) error {
	return e.PredictDense(vals, nrows, ncols, predictions, e.NEstimators())
}

func (e *BornEngine) PredictLeafIndicesCSR(indptr, cols []int, vals, predictions []float64) error {
	return e.PredictCSR(indptr, cols, vals, predictions, e.NEstimators())
}

func (e *BornEngine) predictLeafIndicesRow(fvals []float64, nEst int, predictions []float64, start int) error {
	f := e.forest
	nResults := f.NumOutputGroups * nEst
	for k := 0; k < nResults; k++ {
		predictions[start+k] = 0
	}
	for i := 0; i < nEst; i++ {
		for k := 0; k < f.NumOutputGroups; k++ {
			treeIdx := i*f.NumOutputGroups + k
			if treeIdx >= len(f.Trees) {
				continue
			}
			t := &f.Trees[treeIdx]
			var leaf int32
			if treeNeedsBornWalk(t) {
				leaf = walkTree(t, fvals)
			} else {
				leaf = e.bornWalkTreeBatch1(fvals, t)
			}
			leafIdx := 0.0
			if leaf < 0 {
				leafIdx = float64(int(^leaf))
			}
			predictions[start+k*nEst+i] = leafIdx
		}
	}
	return nil
}

func (e *BornEngine) bornWalkTreeBatch1(fvals []float64, t *TreeIR) int32 {
	return e.bornWalkTreeBatch(fvals, 1, len(fvals), t)[0]
}

func (e *BornEngine) applyTransform(predictions []float64, startIndex int) {
	if e.transform == nil || e.outputType == TransformRaw {
		return
	}
	raw := predictions[startIndex : startIndex+e.nRawOutputGroups]
	e.transform(raw, predictions, startIndex)
}
