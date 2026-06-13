package model

import (
	"fmt"
	"math"

	"github.com/dmitryikh/leaves/explain"
	"github.com/dmitryikh/leaves/predict"
)

func isContribOutput(o predict.OutputKind) bool {
	switch o {
	case predict.OutputContribution, predict.OutputApproxContribution, predict.OutputInteraction:
		return true
	default:
		return false
	}
}

func contribOutputLen(nRows, nFeatures, nGroups int, output predict.OutputKind) int {
	if nGroups <= 0 {
		nGroups = 1
	}
	cols := nFeatures + 1 // 末列 bias，对齐 XGBoost pred_contribs
	switch output {
	case predict.OutputContribution, predict.OutputApproxContribution:
		return nRows * nGroups * cols
	case predict.OutputInteraction:
		return nRows * nGroups * cols * cols
	default:
		return 0
	}
}

func (e *Ensemble) predictContrib(req predict.Request, out []float64, nRows, nFeatures, nGroups int) error {
	f := e.Forest()
	if f == nil {
		return fmt.Errorf("contributions require tree model (gbtree/dart)")
	}
	feats, err := featuresFromRequest(req, nRows, nFeatures)
	if err != nil {
		return err
	}
	expl := explain.NewTreeExplainer(f)
	bases := expl.ExpectedValues()
	if len(bases) == 0 {
		bases = []float64{0}
	}

	switch req.Output {
	case predict.OutputContribution:
		return packContribFromSHAP(expl, feats, bases, nGroups, nFeatures, out)
	case predict.OutputApproxContribution:
		return packContribFromSaabas(expl, feats, bases, nGroups, nFeatures, out)
	case predict.OutputInteraction:
		return packInteractionFromSHAP(expl, feats, bases, nGroups, nFeatures, out)
	default:
		return fmt.Errorf("unsupported contribution output %d", req.Output)
	}
}

func packContribFromSHAP(
	expl *explain.TreeExplainer,
	feats [][]float64,
	bases []float64,
	nGroups, nFeatures int,
	out []float64,
) error {
	cols := nFeatures + 1
	biasIdx := nFeatures
	if nGroups > 1 {
		phi, err := expl.ShapleyValuesMulticlass(feats)
		if err != nil {
			return err
		}
		for si, perClass := range phi {
			for k := 0; k < nGroups; k++ {
				off := si*nGroups*cols + k*cols
				if k < len(perClass) {
					copy(out[off:off+nFeatures], perClass[k])
				}
				base := 0.0
				if k < len(bases) {
					base = bases[k]
				}
				out[off+biasIdx] = base
			}
		}
		return nil
	}
	phi, err := expl.ShapleyValues(feats)
	if err != nil {
		return err
	}
	for si, row := range phi {
		off := si * cols
		copy(out[off:off+nFeatures], row)
		out[off+biasIdx] = bases[0]
	}
	return nil
}

func packContribFromSaabas(
	expl *explain.TreeExplainer,
	feats [][]float64,
	bases []float64,
	nGroups, nFeatures int,
	out []float64,
) error {
	cols := nFeatures + 1
	biasIdx := nFeatures
	if nGroups > 1 {
		phi, err := expl.ApproximateContributionsMulticlass(feats)
		if err != nil {
			return err
		}
		for si, perClass := range phi {
			for k := 0; k < nGroups; k++ {
				off := si*nGroups*cols + k*cols
				if k < len(perClass) {
					copy(out[off:off+nFeatures], perClass[k])
				}
				base := 0.0
				if k < len(bases) {
					base = bases[k]
				}
				out[off+biasIdx] = base
			}
		}
		return nil
	}
	phi, err := expl.ApproximateContributions(feats)
	if err != nil {
		return err
	}
	for si, row := range phi {
		off := si * cols
		copy(out[off:off+nFeatures], row)
		out[off+biasIdx] = bases[0]
	}
	return nil
}

func packInteractionFromSHAP(
	expl *explain.TreeExplainer,
	feats [][]float64,
	bases []float64,
	nGroups, nFeatures int,
	out []float64,
) error {
	cols := nFeatures + 1
	biasIdx := nFeatures
	if nGroups > 1 {
		intr, err := expl.InteractionValuesMulticlass(feats)
		if err != nil {
			return err
		}
		for si, perClass := range intr {
			for k := 0; k < nGroups; k++ {
				off := si*nGroups*cols*cols + k*cols*cols
				if k < len(perClass) {
					copyInteractionBlock(out[off:off+cols*cols], perClass[k], nFeatures)
				}
				base := 0.0
				if k < len(bases) {
					base = bases[k]
				}
				out[off+biasIdx*cols+biasIdx] = base
			}
		}
		return nil
	}
	intr, err := expl.InteractionValues(feats)
	if err != nil {
		return err
	}
	for si, mat := range intr {
		off := si * cols * cols
		copyInteractionBlock(out[off:off+cols*cols], mat, nFeatures)
		out[off+biasIdx*cols+biasIdx] = bases[0]
	}
	return nil
}

func copyInteractionBlock(dst []float64, mat [][]float64, nFeatures int) {
	cols := nFeatures + 1
	for fi := 0; fi < nFeatures; fi++ {
		if fi >= len(mat) {
			break
		}
		for fj := 0; fj < nFeatures; fj++ {
			if fj < len(mat[fi]) {
				dst[fi*cols+fj] = mat[fi][fj]
			}
		}
	}
}

func featuresFromRequest(req predict.Request, nRows, nFeatures int) ([][]float64, error) {
	rows := make([][]float64, nRows)
	switch m := req.Matrix.(type) {
	case predict.DenseMatrix:
		if m.Cols < nFeatures {
			return nil, fmt.Errorf("matrix cols %d < model features %d", m.Cols, nFeatures)
		}
		for i := 0; i < nRows; i++ {
			row := make([]float64, nFeatures)
			base := i * m.Cols
			copy(row, m.Values[base:base+nFeatures])
			rows[i] = row
		}
	case predict.CSRMatrix:
		for i := 0; i < nRows; i++ {
			row := make([]float64, nFeatures)
			for j := range row {
				row[j] = math.NaN()
			}
			start := m.Indptr[i]
			end := m.Indptr[i+1]
			for j := start; j < end; j++ {
				c := m.Cols[j]
				if c >= 0 && c < nFeatures {
					row[c] = m.Values[j]
				}
			}
			rows[i] = row
		}
	default:
		return nil, fmt.Errorf("unsupported matrix kind")
	}
	return rows, nil
}
