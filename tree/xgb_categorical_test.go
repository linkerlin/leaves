package tree

import (
	"testing"
)

func TestApplyXGBCategoricalSplits(t *testing.T) {
	nodes := []LgNodeData{
		{Feature: 0, Threshold: 0, Flags: flagLeftLeaf | flagRightLeaf, Left: 0, Right: 1},
	}
	tir := BuildTreeIR(nodes, []float64{1.0, -1.0}, nil, nil, 0)

	ApplyXGBCategoricalSplits(
		tir,
		[]int{1},
		[]int32{0, 2},
		[]int32{0},
		[]int64{0},
		[]int64{2},
	)
	if err := ValidateXGBCategoricalNode(tir, 0); err != nil {
		t.Fatal(err)
	}
	if !XGBCategoricalGoLeft(tir, 0, 0) {
		t.Error("cat 0 should go left")
	}
	if XGBCategoricalGoLeft(tir, 0, 1) {
		t.Error("cat 1 should go right")
	}
	if !XGBCategoricalGoLeft(tir, 0, 2) {
		t.Error("cat 2 should go left")
	}
}
