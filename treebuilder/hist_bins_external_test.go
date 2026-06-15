package treebuilder

import (
	"testing"

	"github.com/dmitryikh/leaves/data"
)

func TestBuildGlobalHistBinsExternalMatchesDense(t *testing.T) {
	vals := []float64{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}
	labels := []float64{0, 1, 2, 3}
	dm, err := data.NewDense(vals, 4, 3, labels, nil)
	if err != nil {
		t.Fatal(err)
	}
	bm, err := data.SplitDense(dm, 2)
	if err != nil {
		t.Fatal(err)
	}
	want := BuildGlobalHistBins(dm, 8, nil)
	got := BuildGlobalHistBinsExternal(bm, 8, nil)
	for f := 0; f < dm.NumCol(); f++ {
		wc, wn, wok := want.Lookup(f)
		gc, gn, gok := got.Lookup(f)
		if wok != gok || wn != gn || len(wc) != len(gc) {
			t.Fatalf("feat %d: want ok=%v n=%d got ok=%v n=%d", f, wok, wn, gok, gn)
		}
	}
}
