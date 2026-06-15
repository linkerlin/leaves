package data_test

import (
	"testing"

	"github.com/linkerlin/leaves/data"
)

// stubExternal 验证 ExternalMemoryMatrix 接口可实现（T5 接线前编译门禁）。
type stubExternal struct {
	*data.Dense
	batches int
}

func (s *stubExternal) NumBatches() int { return s.batches }

func (s *stubExternal) Batch(b int) (*data.ExternalBatch, error) {
	if b != 0 {
		return nil, nil
	}
	return &data.ExternalBatch{
		Rows:   s.NumRow(),
		Cols:   s.NumCol(),
		Labels: s.Labels(),
		RowAt: func(row int, buf []float64) error {
			return s.Dense.Row(row, buf)
		},
	}, nil
}

func TestExternalMemoryMatrixStub(t *testing.T) {
	vals := []float64{1, 2, 3, 4}
	labels := []float64{0, 1}
	dm, err := data.NewDense(vals, 2, 2, labels, nil)
	if err != nil {
		t.Fatal(err)
	}
	var em data.ExternalMemoryMatrix = &stubExternal{Dense: dm, batches: 1}
	if em.NumBatches() != 1 {
		t.Fatal(em.NumBatches())
	}
	b, err := em.Batch(0)
	if err != nil || b.Rows != 2 {
		t.Fatalf("batch: %v err=%v", b, err)
	}
	buf := make([]float64, 2)
	if err := b.RowAt(1, buf); err != nil || buf[0] != 3 {
		t.Fatalf("rowAt: %v err=%v", buf, err)
	}
}
