//go:build windows

package tree

import (
	borncpu "github.com/born-ml/born/backend/cpu"
	bornwebgpu "github.com/born-ml/born/backend/webgpu"
)

func (e *BornEngine) bornMargins(vals []float64, rows, cols, nEst int) ([][]float64, error) {
	if e.usingGPU {
		if g, ok := e.gpu.(*bornwebgpu.Backend); ok && g != nil {
			return bornForestMarginsDenseGPU(g, e.forest, vals, rows, cols, nEst)
		}
	}
	if e.cpu == nil {
		e.cpu = borncpu.New()
	}
	return bornForestMarginsDense(e.cpu, e.forest, vals, rows, cols, nEst)
}

func (e *BornEngine) bornWalkTreeBatch(fvals []float64, rows, cols int, t *TreeIR) []int32 {
	if e.usingGPU {
		if g, ok := e.gpu.(*bornwebgpu.Backend); ok && g != nil {
			valsF32 := f64SliceToF32(fvals)
			feats := bornDenseFeaturesF32(g, valsF32, rows, cols)
			return walkTreeBatchF32(g, feats, t).Data()
		}
	}
	if e.cpu == nil {
		e.cpu = borncpu.New()
	}
	feats := bornDenseFeatures(e.cpu, fvals, rows, cols)
	return walkTreeBatch(e.cpu, feats, t).Data()
}
