package explain

import (
	"math"
	"testing"

	"github.com/dmitryikh/leaves/tree"
)

const (
	mcFlagLeftLeaf  = 1 << 2
	mcFlagRightLeaf = 1 << 3
)

func TestMulticlassSHAPAdditivitySynthetic(t *testing.T) {
	nodes0 := []tree.LgNodeData{
		{Feature: 0, Threshold: 0.5, Flags: mcFlagLeftLeaf | mcFlagRightLeaf, Left: 0, Right: 1},
	}
	nodes1 := []tree.LgNodeData{
		{Feature: 1, Threshold: 0.5, Flags: mcFlagLeftLeaf | mcFlagRightLeaf, Left: 0, Right: 1},
	}
	t0 := tree.BuildTreeIR(nodes0, []float64{1.0, -1.0}, nil, nil, 0)
	t1 := tree.BuildTreeIR(nodes1, []float64{2.0, -2.0}, nil, nil, 0)
	forest := &tree.ForestIR{
		NumFeatures:     2,
		NumOutputGroups: 2,
		Trees:           []tree.TreeIR{*t0, *t1},
		TreeInfo:        []int{0, 1},
	}

	expl := NewTreeExplainer(forest)
	x := [][]float64{{1.0, 1.0}}
	phi, err := expl.ShapleyValuesMulticlass(x)
	if err != nil {
		t.Fatal(err)
	}
	bases := expl.ExpectedValues()
	margins := predictForestMargins(forest, x[0], 0)
	for k := 0; k < 2; k++ {
		sum := bases[k]
		for _, v := range phi[0][k] {
			sum += v
		}
		if math.Abs(sum-margins[k]) > 1e-6 {
			t.Errorf("class %d: base+shap=%f margin=%f", k, sum, margins[k])
		}
	}
}

func TestMulticlassInteractionRowSum(t *testing.T) {
	nodes0 := []tree.LgNodeData{
		{Feature: 0, Threshold: 0.5, Flags: mcFlagLeftLeaf | mcFlagRightLeaf, Left: 0, Right: 1},
	}
	nodes1 := []tree.LgNodeData{
		{Feature: 0, Threshold: 0.5, Flags: mcFlagLeftLeaf | mcFlagRightLeaf, Left: 0, Right: 1},
	}
	t0 := tree.BuildTreeIR(nodes0, []float64{1.0, -1.0}, nil, nil, 0)
	t1 := tree.BuildTreeIR(nodes1, []float64{0.5, -0.5}, nil, nil, 0)
	forest := &tree.ForestIR{
		NumFeatures:     1,
		NumOutputGroups: 2,
		Trees:           []tree.TreeIR{*t0, *t1},
		TreeInfo:        []int{0, 1},
	}
	expl := NewTreeExplainer(forest)
	x := [][]float64{{1.0}}
	main, err := expl.ShapleyValuesMulticlass(x)
	if err != nil {
		t.Fatal(err)
	}
	intr, err := expl.InteractionValuesMulticlass(x)
	if err != nil {
		t.Fatal(err)
	}
	for k := 0; k < 2; k++ {
		for fi := 0; fi < 1; fi++ {
			rowSum := intr[0][k][fi][fi]
			for fj := 0; fj < 1; fj++ {
				if fi != fj {
					rowSum += intr[0][k][fi][fj]
				}
			}
			if math.Abs(rowSum-main[0][k][fi]) > 1e-6 {
				t.Errorf("class %d feature %d: row sum %f main %f", k, fi, rowSum, main[0][k][fi])
			}
		}
	}
}
