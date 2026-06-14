//go:build js

package tree

import "fmt"

// BornEngine WASM/js 下委托 NativeEngine（Born 不可用）。
type BornEngine struct {
	native *NativeEngine
}

// BornConfig Born 引擎配置（js 忽略 GPU）。
type BornConfig struct {
	UseGPU bool
}

// NewBornEngine 创建引擎；js 平台回退 Native。
func NewBornEngine(forest *ForestIR, transform TransformFn, outputType TransformType, nOutputGroups int, _ *BornConfig) (*BornEngine, error) {
	if forest == nil {
		return nil, fmt.Errorf("born: nil forest")
	}
	return &BornEngine{native: NewNativeEngine(forest, transform, outputType, nOutputGroups)}, nil
}

func (e *BornEngine) BornUsingGPU() bool { return false }

func (e *BornEngine) NOutputGroups() int    { return e.native.NOutputGroups() }
func (e *BornEngine) NRawOutputGroups() int { return e.native.NRawOutputGroups() }
func (e *BornEngine) NFeatures() int        { return e.native.NFeatures() }
func (e *BornEngine) NEstimators() int      { return e.native.NEstimators() }
func (e *BornEngine) NLeaves() []int        { return e.native.NLeaves() }
func (e *BornEngine) Name() string          { return e.native.Name() }
func (e *BornEngine) Forest() *ForestIR     { return e.native.Forest() }
func (e *BornEngine) Close() error          { return e.native.Close() }

func (e *BornEngine) PredictSingle(fvals []float64, nEstimators int) float64 {
	return e.native.PredictSingle(fvals, nEstimators)
}

func (e *BornEngine) Predict(fvals []float64, nEstimators int, predictions []float64) error {
	return e.native.Predict(fvals, nEstimators, predictions)
}

func (e *BornEngine) PredictDense(vals []float64, nrows, ncols int, predictions []float64, nEstimators int) error {
	return e.native.PredictDense(vals, nrows, ncols, predictions, nEstimators)
}

func (e *BornEngine) PredictCSR(indptr, cols []int, vals, predictions []float64, nEstimators int) error {
	return e.native.PredictCSR(indptr, cols, vals, predictions, nEstimators)
}

func (e *BornEngine) PredictLeafIndicesDense(vals []float64, nrows, ncols int, predictions []float64) error {
	return e.native.PredictLeafIndicesDense(vals, nrows, ncols, predictions)
}

func (e *BornEngine) PredictLeafIndicesCSR(indptr, cols []int, vals, predictions []float64) error {
	return e.native.PredictLeafIndicesCSR(indptr, cols, vals, predictions)
}

// BornWebGPUAvailable js 无 WebGPU Born 路径。
func BornWebGPUAvailable() bool { return false }

func bornOpenWebGPU() (any, error) { return nil, fmt.Errorf("born: webgpu unavailable on js") }

func bornCloseWebGPU(any) {}
