// Package data 提供训练期数据容器。
package data

import "fmt"

// Matrix 训练数据矩阵（Dense MVP）。
type Matrix interface {
	NumRow() int
	NumCol() int
	Row(row int, buf []float64) error
	Labels() []float64
	Weights() []float64
}

// Dense 行优先 Dense 矩阵。
type Dense struct {
	Data         []float64
	Rows         int
	Cols         int
	Y            []float64
	W            []float64
	FT     []FeatureType
	FNames []string
}

// NewDense 创建 Dense 矩阵；vals 行优先 [rows*cols]。
func NewDense(vals []float64, rows, cols int, labels []float64, weights []float64) (*Dense, error) {
	if rows <= 0 || cols <= 0 {
		return nil, fmt.Errorf("data: invalid shape %dx%d", rows, cols)
	}
	if len(vals) != rows*cols {
		return nil, fmt.Errorf("data: vals len %d != %d", len(vals), rows*cols)
	}
	if len(labels) != rows {
		return nil, fmt.Errorf("data: labels len %d != rows %d", len(labels), rows)
	}
	if weights != nil && len(weights) != rows {
		return nil, fmt.Errorf("data: weights len %d != rows %d", len(weights), rows)
	}
	return &Dense{Data: vals, Rows: rows, Cols: cols, Y: labels, W: weights}, nil
}

func (d *Dense) NumRow() int { return d.Rows }
func (d *Dense) NumCol() int { return d.Cols }

func (d *Dense) Row(row int, buf []float64) error {
	if row < 0 || row >= d.Rows {
		return fmt.Errorf("data: row %d out of range", row)
	}
	if cap(buf) < d.Cols {
		buf = make([]float64, d.Cols)
	} else {
		buf = buf[:d.Cols]
	}
	off := row * d.Cols
	copy(buf, d.Data[off:off+d.Cols])
	return nil
}

// RowSlice 返回行视图（只读）。
func (d *Dense) RowSlice(row int) []float64 {
	off := row * d.Cols
	return d.Data[off : off+d.Cols]
}

func (d *Dense) Labels() []float64  { return d.Y }
func (d *Dense) Weights() []float64 { return d.W }

func (d *Dense) FeatureTypes() []FeatureType { return d.FT }
func (d *Dense) FeatureNames() []string { return d.FNames }

// WeightAt 返回样本权重，缺省 1。
func WeightAt(dm Matrix, i int) float64 {
	w := dm.Weights()
	if w == nil || i >= len(w) {
		return 1.0
	}
	return w[i]
}
