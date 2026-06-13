package model

import (
	"fmt"

	"github.com/dmitryikh/leaves/predict"
)

// marginEngine 支持 raw margin 预测的后端（跳过 objective transform）。
type marginEngine interface {
	PredictMarginDense(vals []float64, nrows, ncols int, predictions []float64, nEstimators int) error
	PredictMarginCSR(indptr []int, cols []int, vals []float64, predictions []float64, nEstimators int) error
}

// PredictWithRequest 按统一 Request 语义执行批量预测。
func (e *Ensemble) PredictWithRequest(req predict.Request, out []float64) error {
	nRows, nCols, err := matrixShape(req.Matrix)
	if err != nil {
		return err
	}

	nOut := e.NOutputGroups()
	if req.Output == predict.OutputMargin {
		nOut = e.NRawOutputGroups()
	}
	if len(out) < nOut*nRows {
		return fmt.Errorf("output slice too short (need %d)", nOut*nRows)
	}

	nEst := nEstimatorsFromRequest(e, req)

	if isContribOutput(req.Output) {
		f := e.Forest()
		if f == nil {
			return fmt.Errorf("output %d requires tree model", req.Output)
		}
		nGroups := e.NRawOutputGroups()
		if nGroups <= 0 {
			nGroups = 1
		}
		nFeat := e.NFeatures()
		need := contribOutputLen(nRows, nFeat, nGroups, req.Output)
		if len(out) < need {
			return fmt.Errorf("output slice too short (need %d)", need)
		}
		return e.predictContrib(req, out, nRows, nFeat, nGroups)
	}

	switch req.Output {
	case predict.OutputLeaf:
		return e.predictLeaf(req, out, nRows, nCols, nEst)
	case predict.OutputMargin:
		return e.predictMargin(req, out, nRows, nCols, nEst)
	default:
		return e.predictValue(req, out, nRows, nCols, nEst)
	}
}

func (e *Ensemble) predictValue(req predict.Request, out []float64, nRows, nCols, nEst int) error {
	switch m := req.Matrix.(type) {
	case predict.DenseMatrix:
		return e.PredictDense(m.Values, nRows, nCols, out, nEst, 0)
	case predict.CSRMatrix:
		return e.PredictCSR(m.Indptr, m.Cols, m.Values, out, nEst, 0)
	default:
		return fmt.Errorf("unsupported matrix kind")
	}
}

func (e *Ensemble) predictMargin(req predict.Request, out []float64, nRows, nCols, nEst int) error {
	me, ok := e.engine.(marginEngine)
	if !ok {
		return fmt.Errorf("engine does not support margin prediction")
	}
	switch m := req.Matrix.(type) {
	case predict.DenseMatrix:
		return me.PredictMarginDense(m.Values, nRows, nCols, out, nEst)
	case predict.CSRMatrix:
		return me.PredictMarginCSR(m.Indptr, m.Cols, m.Values, out, nEst)
	default:
		return fmt.Errorf("unsupported matrix kind")
	}
}

func (e *Ensemble) predictLeaf(req predict.Request, out []float64, nRows, nCols, nEst int) error {
	switch m := req.Matrix.(type) {
	case predict.DenseMatrix:
		return e.engine.PredictLeafIndicesDense(m.Values, nRows, nCols, out)
	case predict.CSRMatrix:
		return e.engine.PredictLeafIndicesCSR(m.Indptr, m.Cols, m.Values, out)
	default:
		return fmt.Errorf("unsupported matrix kind")
	}
}

func matrixShape(m predict.Matrix) (nRows, nCols int, err error) {
	switch dm := m.(type) {
	case predict.DenseMatrix:
		if dm.Rows <= 0 || dm.Cols <= 0 {
			return 0, 0, fmt.Errorf("invalid dense matrix shape")
		}
		return dm.Rows, dm.Cols, nil
	case predict.CSRMatrix:
		nRows = len(dm.Indptr) - 1
		if nRows < 0 {
			return 0, 0, fmt.Errorf("invalid csr indptr")
		}
		return nRows, eColsFromCSR(dm), nil
	default:
		return 0, 0, fmt.Errorf("unknown matrix type")
	}
}

func eColsFromCSR(m predict.CSRMatrix) int {
	maxCol := 0
	for _, c := range m.Cols {
		if c+1 > maxCol {
			maxCol = c + 1
		}
	}
	return maxCol
}

func nEstimatorsFromRequest(e *Ensemble, req predict.Request) int {
	if req.NEstimators > 0 {
		return req.NEstimators
	}
	if req.IterRange.End > req.IterRange.Begin {
		return req.IterRange.End - req.IterRange.Begin
	}
	return 0
}
