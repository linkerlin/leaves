package io_test

import (
	"bytes"
	"math"
	"testing"

	"github.com/linkerlin/leaves/io"
	"github.com/linkerlin/leaves/model"
	"github.com/linkerlin/leaves/tree"
)

func TestLeavesJSONRoundTrip(t *testing.T) {
	tir := tree.BuildTreeIR(nil, []float64{0.5}, nil, nil, 0)
	ir := &model.ModelIR{
		Kind:             model.KindGBTree,
		NumFeatures:      1,
		NRawOutputGroups: 1,
		NOutputGroups:    1,
		Name:             "leaves.gbtree",
		Forest: &tree.ForestIR{
			NumFeatures:       1,
			NumOutputGroups:   1,
			BaseScore:         0.1,
			Trees:             []tree.TreeIR{*tir},
			IterationIndptr:   []int{0, 1},
			TreeInfo:          []int{0},
			WeightDrop:        []float64{1},
			Name:              "leaves.gbtree",
		},
	}
	var buf bytes.Buffer
	if err := io.SaveLeavesJSON(&buf, ir, "reg:squarederror"); err != nil {
		t.Fatal(err)
	}
	got, obj, err := io.ParseLeavesJSON(bytes.NewReader(buf.Bytes()))
	if err != nil {
		t.Fatal(err)
	}
	if obj != "reg:squarederror" {
		t.Fatalf("objective %q", obj)
	}
	x := []float64{0.0}
	if math.Abs(tree.ForestMargin(got.Forest, x, 0)-0.6) > 1e-9 {
		t.Fatalf("margin %f", tree.ForestMargin(got.Forest, x, 0))
	}
}
