package train_test

import (
	"math"
	"testing"

	"github.com/dmitryikh/leaves/data"
	"github.com/dmitryikh/leaves/train"
	"github.com/dmitryikh/leaves/tree"
)

func synthLargeDense(rows, cols int) data.Matrix {
	vals := make([]float64, rows*cols)
	labels := make([]float64, rows)
	for i := 0; i < rows; i++ {
		labels[i] = float64(i % 5)
		for j := 0; j < cols; j++ {
			vals[i*cols+j] = float64(i)*0.01 + float64(j)*0.17 + float64((i+j)%11)
		}
	}
	dm, _ := data.NewDense(vals, rows, cols, labels, nil)
	return dm
}

func TestGPUMarginPredictParity(t *testing.T) {
	if !tree.BornWebGPUAvailable() {
		t.Skip("webgpu unavailable")
	}
	dm := synthLargeDense(400, 12)
	learner, err := train.NewLearner(train.Config{
		Objective:    train.ObjectiveSquaredError,
		NumRound:     8,
		MaxDepth:     4,
		LearningRate: 0.3,
		TreeMethod:   train.TreeMethodGPUHist,
		AccelMode:    train.AccelModeWebGPU,
		Seed:         7,
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := learner.Fit(dm); err != nil {
		t.Fatal(err)
	}
	m := learner.Model()
	if m == nil || m.Forest == nil {
		t.Fatal("nil model forest")
	}

	d := dm.(*data.Dense)
	cpuOut := make([]float64, d.Rows)
	gpuOut := make([]float64, d.Rows)

	for i := 0; i < d.Rows; i++ {
		row := d.RowSlice(i)
		margin := tree.ForestMargins(m.Forest, row, 0)
		if len(margin) > 0 {
			cpuOut[i] = margin[0]
		}
	}

	eng, err := tree.NewBornEngine(m.Forest, tree.ApplyTransformRaw, tree.TransformRaw, 1, &tree.BornConfig{UseGPU: true})
	if err != nil || !eng.BornUsingGPU() {
		t.Skip("born gpu engine unavailable")
	}
	defer eng.Close()
	if err := eng.PredictDense(d.Data, d.Rows, d.Cols, gpuOut, 0); err != nil {
		t.Fatal(err)
	}

	for i := range cpuOut {
		if math.Abs(cpuOut[i]-gpuOut[i]) > 5e-2 {
			t.Fatalf("margin[%d] cpu=%v gpu=%v", i, cpuOut[i], gpuOut[i])
		}
	}
}

func TestLearnerPredictMarginsUsesGPUPath(t *testing.T) {
	if !tree.BornWebGPUAvailable() {
		t.Skip("webgpu unavailable")
	}
	dm := synthLargeDense(300, 8)
	learner, err := train.NewLearner(train.Config{
		Objective:  train.ObjectiveSquaredError,
		NumRound:   4,
		MaxDepth:   3,
		TreeMethod: train.TreeMethodGPUHist,
		AccelMode:  train.AccelModeWebGPU,
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := learner.Fit(dm); err != nil {
		t.Fatal(err)
	}
	out := make([]float64, 300)
	if err := learner.PredictMargins(dm, out); err != nil {
		t.Fatal(err)
	}
}
