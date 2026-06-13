package train_test

import (
	"fmt"
	"math"
	"strings"
	"testing"

	"github.com/dmitryikh/leaves/data"
	"github.com/dmitryikh/leaves/train"
)

func TestExponentialLRScheduler(t *testing.T) {
	s := train.ExponentialLRScheduler(0.5)
	if math.Abs(s.Rate(0, 0.3)-0.3) > 1e-12 {
		t.Fatalf("round0: %v", s.Rate(0, 0.3))
	}
	if math.Abs(s.Rate(2, 0.3)-0.075) > 1e-12 {
		t.Fatalf("round2: %v", s.Rate(2, 0.3))
	}
}

func TestStepLRScheduler(t *testing.T) {
	s := train.StepLRScheduler(3, 0.5)
	if math.Abs(s.Rate(2, 0.2)-0.2) > 1e-12 {
		t.Fatal("before step")
	}
	if math.Abs(s.Rate(3, 0.2)-0.1) > 1e-12 {
		t.Fatalf("after step: %v", s.Rate(3, 0.2))
	}
}

func TestCallbackRecordsRounds(t *testing.T) {
	vals := []float64{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
	labels := make([]float64, len(vals))
	for i := range labels {
		labels[i] = float64(i)
	}
	dm, _ := data.NewDense(vals, len(vals), 1, labels, nil)

	var seen []int
	var lrs []float64
	cb := train.FuncCallback(func(ctx *train.CallbackContext) error {
		seen = append(seen, ctx.Round)
		lrs = append(lrs, ctx.LearningRate)
		return nil
	})

	learner, err := train.NewLearner(train.Config{
		Objective:    train.ObjectiveSquaredError,
		NumRound:     4,
		MaxDepth:     2,
		LearningRate: 0.4,
		TreeMethod:   train.TreeMethodExact,
		LRScheduler:  train.ExponentialLRScheduler(0.5),
		Callbacks:    []train.TrainingCallback{cb},
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := learner.Fit(dm); err != nil {
		t.Fatal(err)
	}
	if len(seen) != 4 {
		t.Fatalf("callbacks: %v", seen)
	}
	if math.Abs(lrs[0]-0.4) > 1e-12 || math.Abs(lrs[1]-0.2) > 1e-12 {
		t.Fatalf("lr decay: %v", lrs)
	}
}

func TestCallbackEvalMetric(t *testing.T) {
	vals := make([]float64, 40)
	labels := make([]float64, 20)
	for i := 0; i < 20; i++ {
		vals[i*2] = float64(i)
		vals[i*2+1] = float64(i % 5)
		labels[i] = float64(i % 2)
	}
	trainDM, _ := data.NewDense(vals, 20, 2, labels, nil)

	valLabels := []float64{1, 0, 1, 0, 1, 0, 1, 0, 0, 1}
	valVals := make([]float64, 20)
	for i := 0; i < 10; i++ {
		valVals[i*2] = float64(i + 100)
		valVals[i*2+1] = float64(i % 3)
	}
	valDM, _ := data.NewDense(valVals, 10, 2, valLabels, nil)

	var evalOK int
	cb := train.FuncCallback(func(ctx *train.CallbackContext) error {
		if ctx.EvalMetricOK {
			evalOK++
		}
		return nil
	})

	learner, err := train.NewLearner(train.Config{
		Objective:    train.ObjectiveBinaryLogistic,
		EvalMetric:   train.EvalAUC,
		NumRound:     3,
		MaxDepth:     2,
		LearningRate: 0.3,
		TreeMethod:   train.TreeMethodExact,
		EvalSet:      valDM,
		Callbacks:    []train.TrainingCallback{cb},
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := learner.Fit(trainDM); err != nil {
		t.Fatal(err)
	}
	if evalOK != 3 {
		t.Fatalf("expected eval metric each round, got %d", evalOK)
	}
}

func TestCallbackErrorStopsTraining(t *testing.T) {
	vals := []float64{0, 1, 2, 3, 4, 5}
	labels := []float64{0, 1, 0, 1, 0, 1}
	dm, _ := data.NewDense(vals, 6, 1, labels, nil)

	learner, err := train.NewLearner(train.Config{
		Objective:    train.ObjectiveSquaredError,
		NumRound:     5,
		MaxDepth:     2,
		LearningRate: 0.3,
		TreeMethod:   train.TreeMethodExact,
		Callbacks: []train.TrainingCallback{
			train.FuncCallback(func(ctx *train.CallbackContext) error {
				if ctx.Round == 1 {
					return fmt.Errorf("callback abort")
				}
				return nil
			}),
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	err = learner.Fit(dm)
	if err == nil || !strings.Contains(err.Error(), "callback abort") {
		t.Fatalf("expected callback error, got %v", err)
	}
	if len(learner.MetricHistory()) > 2 {
		t.Fatalf("training should stop early, history len=%d", len(learner.MetricHistory()))
	}
}
