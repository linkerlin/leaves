//go:build !js

package tree

import (
	"fmt"

	"github.com/born-ml/born/tensor"
)

// bornForestMarginsDense Born 批量 raw margin，直接读 ForestIR。
func bornForestMarginsDense[B tensor.Backend](
	b B,
	f *ForestIR,
	vals []float64,
	rows, cols, nEst int,
) ([][]float64, error) {
	if f == nil || rows <= 0 {
		return nil, fmt.Errorf("born: invalid forest or rows")
	}
	g := f.NumOutputGroups
	if g <= 0 {
		g = 1
	}
	margins := make([][]float64, rows)
	for i := range margins {
		margins[i] = make([]float64, g)
		for k := 0; k < g; k++ {
			margins[i][k] = classBaseScore(f, k)
		}
	}

	nEstAdj := adjustNEstimators(f, nEst)
	if nEstAdj <= 0 {
		return margins, nil
	}

	feats := bornDenseFeatures(b, vals, rows, cols)

	addTree := func(ti int, coef float64) {
		if ti < 0 || ti >= len(f.Trees) {
			return
		}
		t := &f.Trees[ti]
		w := weightDrop(f, ti) * coef
		k := forestTreeClassIndex(f, ti)

		if treeNeedsBornWalk(t) {
			for r := 0; r < rows; r++ {
				fv := bornRowSlice(vals, r, cols)
				if t.OutputDim > 1 {
					vec := treeVectorMargin(t, fv)
					for d := 0; d < len(vec) && d < g; d++ {
						margins[r][d] += vec[d] * w
					}
				} else if k >= 0 && k < g {
					margins[r][k] += predictTreeScalar(t, fv) * w
				}
			}
			return
		}

		if t.OutputDim > 1 {
			vecs := treeVectorBatch(b, feats, t)
			for r := 0; r < rows; r++ {
				for d := 0; d < len(vecs[r]) && d < g; d++ {
					margins[r][d] += vecs[r][d] * w
				}
			}
			return
		}
		scalars := treeScalarBatch(b, feats, t)
		for r := 0; r < rows; r++ {
			if k >= 0 && k < g {
				margins[r][k] += scalars[r] * w
			}
		}
	}

	if len(f.IterationIndptr) > 1 {
		for iter := 0; iter < nEstAdj; iter++ {
			if iter+1 >= len(f.IterationIndptr) {
				break
			}
			for ti := f.IterationIndptr[iter]; ti < f.IterationIndptr[iter+1]; ti++ {
				addTree(ti, 1.0)
			}
		}
		if f.AverageOutput {
			bornScaleMarginsExceptBase(f, margins, 1.0/float64(nEstAdj))
		}
		return margins, nil
	}

	coef := 1.0
	if f.AverageOutput {
		coef = 1.0 / float64(nEstAdj)
	}
	for i := 0; i < nEstAdj; i++ {
		for k := 0; k < g; k++ {
			treeIdx := i*g + k
			addTree(treeIdx, coef)
		}
	}
	return margins, nil
}

func bornRowSlice(vals []float64, row, cols int) []float64 {
	start := row * cols
	end := start + cols
	if end > len(vals) {
		end = len(vals)
	}
	return vals[start:end]
}

func bornScaleMarginsExceptBase(f *ForestIR, margins [][]float64, scale float64) {
	for i := range margins {
		scaleExceptBase(f, margins[i], scale)
	}
}
