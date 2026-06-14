//go:build !windows && !js

package tree

import (
	"fmt"

	borncpu "github.com/born-ml/born/backend/cpu"
)

func (e *BornEngine) bornMargins(vals []float64, rows, cols, nEst int) ([][]float64, error) {
	if e.cpu == nil {
		e.cpu = borncpu.New()
	}
	return bornForestMarginsDense(e.cpu, e.forest, vals, rows, cols, nEst)
}

func (e *BornEngine) bornWalkTreeBatch(fvals []float64, rows, cols int, t *TreeIR) []int32 {
	if e.cpu == nil {
		e.cpu = borncpu.New()
	}
	feats := bornDenseFeatures(e.cpu, fvals, rows, cols)
	return walkTreeBatch(e.cpu, feats, t).Data()
}

func bornForestMarginsDenseGPU(b any, f *ForestIR, vals []float64, rows, cols, nEst int) ([][]float64, error) {
	_ = b
	_ = f
	return nil, fmt.Errorf("born: gpu margins require windows")
}
