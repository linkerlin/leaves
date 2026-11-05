package data

import "fmt"

// SliceMatrix 按行索引切片矩阵（Dense/CSR）。
func SliceMatrix(dm Matrix, idx []int) (Matrix, error) {
	if dm == nil {
		return nil, fmt.Errorf("data: nil matrix")
	}
	switch m := dm.(type) {
	case *Dense:
		return sliceDense(m, idx)
	case *DenseWithGroups:
		return sliceDense(m.Dense, idx)
	case *CSR:
		return sliceCSR(m, idx)
	default:
		return sliceGeneric(dm, idx)
	}
}

// SliceMatrixGroups 按组索引切片 GroupedMatrix，保持组内样本连续且不跨切分边界。
func SliceMatrixGroups(gm GroupedMatrix, groupIdx []int) (*DenseWithGroups, error) {
	if gm == nil {
		return nil, fmt.Errorf("data: nil grouped matrix")
	}
	groups := gm.Groups()
	offsets := make([]int, len(groups)+1)
	for i, g := range groups {
		offsets[i+1] = offsets[i] + g
	}
	var rowIdx []int
	subGroups := make([]int, 0, len(groupIdx))
	for _, gi := range groupIdx {
		if gi < 0 || gi >= len(groups) {
			return nil, fmt.Errorf("data: group index %d out of range [0,%d)", gi, len(groups))
		}
		for r := offsets[gi]; r < offsets[gi+1]; r++ {
			rowIdx = append(rowIdx, r)
		}
		subGroups = append(subGroups, groups[gi])
	}
	var base Matrix = gm
	if dwg, ok := gm.(*DenseWithGroups); ok {
		base = dwg.Dense
	}
	sliced, err := SliceMatrix(base, rowIdx)
	if err != nil {
		return nil, err
	}
	d, ok := sliced.(*Dense)
	if !ok {
		gen, err := sliceGeneric(sliced, makeRange(sliced.NumRow()))
		if err != nil {
			return nil, err
		}
		d = gen.(*Dense)
	}
	return NewDenseWithGroups(d, subGroups)
}

func makeRange(n int) []int {
	out := make([]int, n)
	for i := range out {
		out[i] = i
	}
	return out
}

func sliceDense(d *Dense, idx []int) (*Dense, error) {
	n := len(idx)
	cols := d.Cols
	vals := make([]float64, n*cols)
	labels := make([]float64, n)
	var weights []float64
	if d.W != nil {
		weights = make([]float64, n)
	}
	var idVals []string
	if len(d.IDValues) == d.Rows {
		idVals = make([]string, n)
	}
	var yMulti []float64
	if d.NTargets > 1 && len(d.YMulti) == d.Rows*d.NTargets {
		yMulti = make([]float64, n*d.NTargets)
	}
	row := make([]float64, cols)
	for i, ri := range idx {
		if err := d.Row(ri, row); err != nil {
			return nil, err
		}
		copy(vals[i*cols:(i+1)*cols], row)
		labels[i] = d.Y[ri]
		if weights != nil {
			weights[i] = dataWeightAt(d.W, ri)
		}
		if idVals != nil {
			idVals[i] = d.IDValues[ri]
		}
		if yMulti != nil {
			copy(yMulti[i*d.NTargets:(i+1)*d.NTargets], d.YMulti[ri*d.NTargets:(ri+1)*d.NTargets])
		}
	}
	out := &Dense{
		Data:         vals,
		Rows:         n,
		Cols:         cols,
		Y:            labels,
		W:            weights,
		MissingLabel: d.MissingLabel,
		LabelName:    d.LabelName,
		IDName:       d.IDName,
		IDValues:     idVals,
		NTargets:     d.NTargets,
		YMulti:       yMulti,
	}
	if len(d.FT) > 0 {
		out.FT = append([]FeatureType(nil), d.FT...)
	}
	if len(d.FNames) > 0 {
		out.FNames = append([]string(nil), d.FNames...)
	}
	if len(d.HeaderCols) > 0 {
		out.HeaderCols = append([]string(nil), d.HeaderCols...)
	}
	if len(d.DroppedCols) > 0 {
		out.DroppedCols = append([]string(nil), d.DroppedCols...)
	}
	if len(d.CategoryMaps) > 0 {
		out.CategoryMaps = d.CategoryMaps
	}
	return out, nil
}

func sliceCSR(c *CSR, idx []int) (*CSR, error) {
	n := len(idx)
	labels := make([]float64, n)
	var weights []float64
	if c.W != nil {
		weights = make([]float64, n)
	}
	indptr := make([]int, n+1)
	var cols []int
	var vals []float64
	for i, ri := range idx {
		start := c.Indptr[ri]
		end := c.Indptr[ri+1]
		indptr[i] = len(cols)
		cols = append(cols, c.Cols[start:end]...)
		vals = append(vals, c.Vals[start:end]...)
		labels[i] = c.Y[ri]
		if weights != nil {
			weights[i] = dataWeightAt(c.W, ri)
		}
	}
	indptr[n] = len(cols)
	return NewCSR(indptr, cols, vals, c.ColsN, labels, weights)
}

func sliceGeneric(dm Matrix, idx []int) (Matrix, error) {
	cols := dm.NumCol()
	n := len(idx)
	vals := make([]float64, n*cols)
	labels := make([]float64, n)
	row := make([]float64, cols)
	for i, ri := range idx {
		if err := dm.Row(ri, row); err != nil {
			return nil, err
		}
		copy(vals[i*cols:(i+1)*cols], row)
		labels[i] = dm.Labels()[ri]
	}
	return NewDense(vals, n, cols, labels, nil)
}

func dataWeightAt(w []float64, i int) float64 {
	if w == nil || i >= len(w) {
		return 1.0
	}
	return w[i]
}
