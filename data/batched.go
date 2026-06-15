package data

import "fmt"

// BatchedMatrix 按批提供特征行，labels/weights 全量在内存（对标 XGBoost ExtMem）。
type BatchedMatrix struct {
	Cols     int
	labels   []float64
	weights  []float64
	Batches  []*ExternalBatch
	rowStart []int
}

// NewBatchedMatrix 从 ExternalBatch 列表构建外存矩阵视图。
func NewBatchedMatrix(batches []*ExternalBatch, labels, weights []float64) (*BatchedMatrix, error) {
	if len(batches) == 0 {
		return nil, fmt.Errorf("data: empty external batches")
	}
	cols := batches[0].Cols
	if cols <= 0 {
		return nil, fmt.Errorf("data: invalid cols")
	}
	total := 0
	for i, b := range batches {
		if b == nil {
			return nil, fmt.Errorf("data: nil batch %d", i)
		}
		if b.Cols != cols {
			return nil, fmt.Errorf("data: batch %d cols %d != %d", i, b.Cols, cols)
		}
		if b.Rows <= 0 {
			return nil, fmt.Errorf("data: batch %d empty", i)
		}
		if b.RowAt == nil {
			return nil, fmt.Errorf("data: batch %d missing RowAt", i)
		}
		if len(b.Labels) > 0 && len(b.Labels) != b.Rows {
			return nil, fmt.Errorf("data: batch %d labels len mismatch", i)
		}
		total += b.Rows
	}
	if len(labels) != total {
		return nil, fmt.Errorf("data: labels len %d != rows %d", len(labels), total)
	}
	if weights != nil && len(weights) != total {
		return nil, fmt.Errorf("data: weights len %d != rows %d", len(weights), total)
	}

	rowStart := make([]int, len(batches))
	off := 0
	for i, b := range batches {
		rowStart[i] = off
		off += b.Rows
	}
	return &BatchedMatrix{
		Cols:     cols,
		labels:   labels,
		weights:  weights,
		Batches:  batches,
		rowStart: rowStart,
	}, nil
}

// SplitDense 将 Dense 切分为固定行数的批次（测试 / 单机模拟外存）。
func SplitDense(d *Dense, batchRows int) (*BatchedMatrix, error) {
	if d == nil {
		return nil, fmt.Errorf("data: nil dense")
	}
	if batchRows <= 0 {
		batchRows = d.Rows
	}
	var batches []*ExternalBatch
	for start := 0; start < d.Rows; start += batchRows {
		end := start + batchRows
		if end > d.Rows {
			end = d.Rows
		}
		n := end - start
		base := start
		batches = append(batches, &ExternalBatch{
			Rows:  n,
			Cols:  d.Cols,
			RowAt: func(row int, buf []float64) error {
				return d.Row(base+row, buf)
			},
		})
	}
	return NewBatchedMatrix(batches, d.Labels(), d.Weights())
}

func (b *BatchedMatrix) NumRow() int {
	if b == nil {
		return 0
	}
	n := 0
	for _, batch := range b.Batches {
		n += batch.Rows
	}
	return n
}

func (b *BatchedMatrix) NumCol() int {
	if b == nil {
		return 0
	}
	return b.Cols
}

func (b *BatchedMatrix) Labels() []float64  { return b.labels }
func (b *BatchedMatrix) Weights() []float64 { return b.weights }

func (b *BatchedMatrix) Row(row int, buf []float64) error {
	if b == nil {
		return fmt.Errorf("data: nil batched matrix")
	}
	bi, local, err := b.locate(row)
	if err != nil {
		return err
	}
	return b.Batches[bi].RowAt(local, buf)
}

func (b *BatchedMatrix) NumBatches() int {
	if b == nil {
		return 0
	}
	return len(b.Batches)
}

func (b *BatchedMatrix) Batch(i int) (*ExternalBatch, error) {
	if b == nil || i < 0 || i >= len(b.Batches) {
		return nil, fmt.Errorf("data: batch %d out of range", i)
	}
	batch := b.Batches[i]
	out := *batch
	if len(batch.Labels) == 0 && len(b.labels) > 0 {
		start := b.rowStart[i]
		end := start + batch.Rows
		out.Labels = b.labels[start:end]
	}
	if len(batch.Weights) == 0 && len(b.weights) > 0 {
		start := b.rowStart[i]
		end := start + batch.Rows
		out.Weights = b.weights[start:end]
	}
	return &out, nil
}

func (b *BatchedMatrix) locate(row int) (batchIdx, localRow int, err error) {
	if row < 0 || row >= b.NumRow() {
		return 0, 0, fmt.Errorf("data: row %d out of range", row)
	}
	for i := len(b.Batches) - 1; i >= 0; i-- {
		if row >= b.rowStart[i] {
			return i, row - b.rowStart[i], nil
		}
	}
	return 0, 0, fmt.Errorf("data: row %d not found", row)
}

// MaterializeExternal 将全部批次物化为 Dense（exact 路径或调试）。
func MaterializeExternal(em ExternalMemoryMatrix) (*Dense, error) {
	if em == nil {
		return nil, fmt.Errorf("data: nil external matrix")
	}
	n := em.NumRow()
	c := em.NumCol()
	vals := make([]float64, n*c)
	row := make([]float64, c)
	for i := 0; i < n; i++ {
		if err := em.Row(i, row); err != nil {
			return nil, err
		}
		copy(vals[i*c:(i+1)*c], row)
	}
	return NewDense(vals, n, c, em.Labels(), em.Weights())
}

// AsExternalMemoryMatrix 若已是外存矩阵则返回，否则 nil。
func AsExternalMemoryMatrix(dm Matrix) ExternalMemoryMatrix {
	em, _ := dm.(ExternalMemoryMatrix)
	return em
}
