package data_test

import (
	"testing"

	"github.com/linkerlin/leaves/data"
)

func TestBatchedMatrixRow(t *testing.T) {
	vals := []float64{1, 2, 3, 4, 5, 6}
	labels := []float64{0, 1, 2}
	dm, err := data.NewDense(vals, 3, 2, labels, nil)
	if err != nil {
		t.Fatal(err)
	}
	bm, err := data.SplitDense(dm, 2)
	if err != nil {
		t.Fatal(err)
	}
	if bm.NumBatches() != 2 {
		t.Fatalf("batches=%d", bm.NumBatches())
	}
	buf := make([]float64, 2)
	if err := bm.Row(2, buf); err != nil || buf[0] != 5 {
		t.Fatalf("row2=%v err=%v", buf, err)
	}
}

func TestMaterializeExternal(t *testing.T) {
	vals := []float64{1, 2, 3, 4}
	labels := []float64{0, 1}
	dm, _ := data.NewDense(vals, 2, 2, labels, nil)
	bm, err := data.SplitDense(dm, 1)
	if err != nil {
		t.Fatal(err)
	}
	got, err := data.MaterializeExternal(bm)
	if err != nil {
		t.Fatal(err)
	}
	if got.NumRow() != 2 || got.Data[2] != 3 {
		t.Fatalf("materialized=%v", got.Data)
	}
}
