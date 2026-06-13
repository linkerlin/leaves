package data

import "fmt"

// CSR 训练期 CSR 矩阵（缺失特征视为 0）。
type CSR struct {
	Indptr []int
	Cols   []int
	Vals   []float64
	ColsN  int
	Y      []float64
	W      []float64
}

// NewCSR 创建 CSR 矩阵。
func NewCSR(indptr, cols []int, vals []float64, numCol int, labels []float64, weights []float64) (*CSR, error) {
	rows := len(indptr) - 1
	if rows < 0 {
		return nil, fmt.Errorf("data: invalid indptr")
	}
	if numCol <= 0 {
		return nil, fmt.Errorf("data: invalid cols %d", numCol)
	}
	if labels != nil && len(labels) != rows {
		return nil, fmt.Errorf("data: labels len %d != rows %d", len(labels), rows)
	}
	if weights != nil && len(weights) != rows {
		return nil, fmt.Errorf("data: weights len %d != rows %d", len(weights), rows)
	}
	return &CSR{
		Indptr: indptr,
		Cols:   cols,
		Vals:   vals,
		ColsN:  numCol,
		Y:      labels,
		W:      weights,
	}, nil
}

func (c *CSR) NumRow() int { return len(c.Indptr) - 1 }
func (c *CSR) NumCol() int { return c.ColsN }

func (c *CSR) Row(row int, buf []float64) error {
	if row < 0 || row >= c.NumRow() {
		return fmt.Errorf("data: row %d out of range", row)
	}
	if cap(buf) < c.ColsN {
		buf = make([]float64, c.ColsN)
	} else {
		buf = buf[:c.ColsN]
		for i := range buf {
			buf[i] = 0
		}
	}
	start := c.Indptr[row]
	end := c.Indptr[row+1]
	for j := start; j < end; j++ {
		col := c.Cols[j]
		if col >= 0 && col < c.ColsN {
			buf[col] = c.Vals[j]
		}
	}
	return nil
}

func (c *CSR) Labels() []float64  { return c.Y }
func (c *CSR) Weights() []float64 { return c.W }

// CSRFromLibsvm 从 libsvm 记录构建 CSR（首列可为 label）。
func CSRFromLibsvm(indptr, cols []int, vals []float64, numCol int, labels []float64) (*CSR, error) {
	return NewCSR(indptr, cols, vals, numCol, labels, nil)
}
