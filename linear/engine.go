package linear

import (
	"fmt"

	"github.com/linkerlin/leaves/predict"
	"github.com/linkerlin/leaves/tree"
)

// 编译期检查：NativeEngine 实现 predict.Engine。
var _ predict.Engine = (*NativeEngine)(nil)

// NativeEngine 纯 Go 线性模型推理引擎。
type NativeEngine struct {
	linear      *LinearIR
	transform   tree.TransformFn
	outputType  tree.TransformType
	nOutputGroups int
}

// NewNativeEngine 创建线性推理引擎。
func NewNativeEngine(lin *LinearIR, transform tree.TransformFn, outputType tree.TransformType, nOutputGroups int) *NativeEngine {
	if nOutputGroups <= 0 {
		nOutputGroups = lin.NumOutputGroups
	}
	return &NativeEngine{
		linear:        lin,
		transform:     transform,
		outputType:    outputType,
		nOutputGroups: nOutputGroups,
	}
}

func (e *NativeEngine) adjNEst(n int) int { return 1 }

func (e *NativeEngine) predictRaw(fvals []float64, predictions []float64, startIndex int) {
	lin := e.linear
	g := lin.NumOutputGroups
	nf := lin.NumFeatures
	biasOff := nf * g

	for k := 0; k < g; k++ {
		sum := lin.BaseScore
		if biasOff+k < len(lin.Weights) {
			sum += lin.Weights[biasOff+k]
		}
		for i := 0; i < nf && i < len(fvals); i++ {
			idx := g*i + k
			if idx < len(lin.Weights) {
				sum += fvals[i] * lin.Weights[idx]
			}
		}
		predictions[startIndex+k] = sum
	}
}

func (e *NativeEngine) applyTransform(predictions []float64, startIndex int) {
	if e.transform == nil || e.outputType == tree.TransformRaw {
		return
	}
	raw := predictions[startIndex : startIndex+e.linear.NumOutputGroups]
	e.transform(raw, predictions, startIndex)
}

func (e *NativeEngine) PredictSingle(fvals []float64, nEstimators int) float64 {
	if e.NOutputGroups() != 1 {
		return 0.0
	}
	if e.NFeatures() > len(fvals) {
		return 0.0
	}
	predictions := make([]float64, 1)
	e.predictRaw(fvals, predictions, 0)
	e.applyTransform(predictions, 0)
	return predictions[0]
}

func (e *NativeEngine) Predict(fvals []float64, nEstimators int, predictions []float64) error {
	if len(predictions) < e.NOutputGroups() {
		return fmt.Errorf("predictions slice too short")
	}
	if e.NFeatures() > len(fvals) {
		return fmt.Errorf("incorrect number of features")
	}
	e.predictRaw(fvals, predictions, 0)
	e.applyTransform(predictions, 0)
	return nil
}

func (e *NativeEngine) PredictDense(
	vals []float64, nrows int, ncols int,
	predictions []float64,
	nEstimators int,
) error {
	if len(predictions) < e.NOutputGroups()*nrows {
		return fmt.Errorf("predictions slice too short")
	}
	if ncols == 0 || e.NFeatures() > ncols {
		return fmt.Errorf("incorrect number of columns")
	}
	g := e.NOutputGroups()
	for i := 0; i < nrows; i++ {
		fvals := vals[i*ncols : (i+1)*ncols]
		start := i * g
		e.predictRaw(fvals, predictions, start)
		e.applyTransform(predictions, start)
	}
	return nil
}

func (e *NativeEngine) PredictMarginDense(
	vals []float64, nrows int, ncols int,
	predictions []float64,
	nEstimators int,
) error {
	if len(predictions) < e.NRawOutputGroups()*nrows {
		return fmt.Errorf("predictions slice too short")
	}
	if ncols == 0 || e.NFeatures() > ncols {
		return fmt.Errorf("incorrect number of columns")
	}
	g := e.NRawOutputGroups()
	for i := 0; i < nrows; i++ {
		fvals := vals[i*ncols : (i+1)*ncols]
		e.predictRaw(fvals, predictions, i*g)
	}
	return nil
}

func (e *NativeEngine) PredictCSR(
	indptr []int, cols []int, vals []float64,
	predictions []float64,
	nEstimators int,
) error {
	nRows := len(indptr) - 1
	if len(predictions) < e.NOutputGroups()*nRows {
		return fmt.Errorf("predictions slice too short")
	}
	g := e.NOutputGroups()
	nf := e.NFeatures()
	for i := 0; i < nRows; i++ {
		fvals := make([]float64, nf)
		start := indptr[i]
		end := indptr[i+1]
		for j := start; j < end; j++ {
			if cols[j] < nf {
				fvals[cols[j]] = vals[j]
			}
		}
		idx := i * g
		e.predictRaw(fvals, predictions, idx)
		e.applyTransform(predictions, idx)
	}
	return nil
}

func (e *NativeEngine) PredictMarginCSR(
	indptr []int, cols []int, vals []float64,
	predictions []float64,
	nEstimators int,
) error {
	nRows := len(indptr) - 1
	if len(predictions) < e.NRawOutputGroups()*nRows {
		return fmt.Errorf("predictions slice too short")
	}
	g := e.NRawOutputGroups()
	nf := e.NFeatures()
	for i := 0; i < nRows; i++ {
		fvals := make([]float64, nf)
		start := indptr[i]
		end := indptr[i+1]
		for j := start; j < end; j++ {
			if cols[j] < nf {
				fvals[cols[j]] = vals[j]
			}
		}
		e.predictRaw(fvals, predictions, i*g)
	}
	return nil
}

func (e *NativeEngine) PredictLeafIndicesDense(
	vals []float64, nrows int, ncols int,
	predictions []float64,
) error {
	return fmt.Errorf("gblinear does not support leaf index prediction")
}

func (e *NativeEngine) PredictLeafIndicesCSR(
	indptr []int, cols []int, vals []float64,
	predictions []float64,
) error {
	return fmt.Errorf("gblinear does not support leaf index prediction")
}

func (e *NativeEngine) NOutputGroups() int   { return e.nOutputGroups }
func (e *NativeEngine) NRawOutputGroups() int { return e.linear.NumOutputGroups }
func (e *NativeEngine) NFeatures() int        { return e.linear.NumFeatures }
func (e *NativeEngine) NEstimators() int      { return 1 }
func (e *NativeEngine) NLeaves() []int        { return nil }
func (e *NativeEngine) Name() string {
	if e.linear.Name != "" {
		return e.linear.Name
	}
	return "xgboost.gblinear"
}
func (e *NativeEngine) Close() error { return nil }
