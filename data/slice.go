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
	case *CSR:
		return sliceCSR(m, idx)
	default:
		return sliceGeneric(dm, idx)
	}
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
	}
	out := &Dense{Data: vals, Rows: n, Cols: cols, Y: labels, W: weights}
	if len(d.FT) > 0 {
		out.FT = append([]FeatureType(nil), d.FT...)
	}
	if len(d.FNames) > 0 {
		out.FNames = append([]string(nil), d.FNames...)
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
