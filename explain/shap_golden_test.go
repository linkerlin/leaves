package explain

import (
	"math"
	"testing"

	"github.com/linkerlin/leaves/tree"
)

// 与 tree/convert.go 中 flag 位一致（仅测试用）。
const (
	testFlagLeftLeaf  = 1 << 2
	testFlagRightLeaf = 1 << 3
)

func TestShapGoldenSingleSplit(t *testing.T) {
	nodes := []tree.LgNodeData{
		{Feature: 0, Threshold: 0.5, Flags: testFlagLeftLeaf | testFlagRightLeaf, Left: 0, Right: 1},
	}
	leafValues := []float64{1.0, -1.0}
	tir := tree.BuildTreeIR(nodes, leafValues, nil, nil, 0)
	forest := &tree.ForestIR{NumFeatures: 1, NumOutputGroups: 1, Trees: []tree.TreeIR{*tir}}

	expl := NewTreeExplainer(forest)
	x := [][]float64{{1.0}}
	phi, err := expl.ShapleyValues(x)
	if err != nil {
		t.Fatal(err)
	}
	want := -2.0
	if math.Abs(phi[0][0]-want) > 1e-9 {
		t.Errorf("phi[0]=%f want %f", phi[0][0], want)
	}
}

func TestInteractionRowSumEqualsMain(t *testing.T) {
	nodes := []tree.LgNodeData{
		{Feature: 0, Threshold: 0.5, Flags: testFlagLeftLeaf | testFlagRightLeaf, Left: 0, Right: 1},
		{Feature: 1, Threshold: 0.5, Flags: testFlagLeftLeaf | testFlagRightLeaf, Left: 2, Right: 3},
	}
	leafValues := []float64{1.0, -1.0, 0.5, -0.5}
	tir := tree.BuildTreeIR(nodes, leafValues, nil, nil, 0)
	forest := &tree.ForestIR{NumFeatures: 2, NumOutputGroups: 1, Trees: []tree.TreeIR{*tir}}

	expl := NewTreeExplainer(forest)
	x := [][]float64{{1.0, 1.0}}
	main, err := expl.ShapleyValues(x)
	if err != nil {
		t.Fatal(err)
	}
	intr, err := expl.InteractionValues(x)
	if err != nil {
		t.Fatal(err)
	}
	for fi := 0; fi < 2; fi++ {
		rowSum := 0.0
		for fj := 0; fj < 2; fj++ {
			rowSum += intr[0][fi][fj]
		}
		if math.Abs(rowSum-main[0][fi]) > 1e-6 {
			t.Errorf("feature %d: row sum %f main %f", fi, rowSum, main[0][fi])
		}
	}
}

func TestInteractionMatrixAdditivity(t *testing.T) {
	nodes := []tree.LgNodeData{
		{Feature: 0, Threshold: 0.5, Flags: testFlagLeftLeaf | testFlagRightLeaf, Left: 0, Right: 1},
	}
	leafValues := []float64{2.0, -1.0}
	tir := tree.BuildTreeIR(nodes, leafValues, nil, nil, 0)
	forest := &tree.ForestIR{NumFeatures: 1, NumOutputGroups: 1, Trees: []tree.TreeIR{*tir}}

	expl := NewTreeExplainer(forest)
	x := [][]float64{{1.0}}
	intr, err := expl.InteractionValues(x)
	if err != nil {
		t.Fatal(err)
	}
	total := expl.ExpectedValue()
	for _, row := range intr[0] {
		for _, v := range row {
			total += v
		}
	}
	margin := predictForestMargin(forest, x[0], 0)
	if math.Abs(total-margin) > 1e-6 {
		t.Errorf("matrix additivity: total=%f margin=%f", total, margin)
	}
}
