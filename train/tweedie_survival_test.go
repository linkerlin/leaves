package train

import (
	"math"
	"testing"

	"github.com/dmitryikh/leaves/data"
)

func TestTweedieTrainSmoke(t *testing.T) {
	vals := []float64{
		1, 0, 0, 1, 1, 1, 0, 0,
		1, 0, 0, 1, 1, 1, 0, 0,
	}
	labels := []float64{0, 1, 2, 0.5, 3, 0, 1.5, 2.5}
	dm, err := data.NewDense(vals, 8, 2, labels, nil)
	if err != nil {
		t.Fatal(err)
	}
	cfg := Config{
		Objective:            ObjectiveTweedie,
		TweedieVariancePower: 1.5,
		NumRound:             3,
		MaxDepth:             2,
		LearningRate:         0.3,
		TreeMethod:           TreeMethodExact,
	}
	learner, err := NewLearner(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := learner.Fit(dm); err != nil {
		t.Fatal(err)
	}
	if learner.Model() == nil || len(learner.Model().Forest.Trees) == 0 {
		t.Fatal("expected trees")
	}
}

func TestSurvivalCoxTrainSmoke(t *testing.T) {
	vals := []float64{
		1, 0, 0, 1, 1, 1, 0, 0,
		1, 0, 0, 1, 1, 1, 0, 0,
	}
	labels := []float64{1, -5, 2, -3, 4, -10, 3, -2}
	dm, err := data.NewDense(vals, 8, 2, labels, nil)
	if err != nil {
		t.Fatal(err)
	}
	cfg := Config{
		Objective:    ObjectiveSurvivalCox,
		NumRound:     2,
		MaxDepth:     2,
		LearningRate: 0.1,
		TreeMethod:   TreeMethodExact,
	}
	learner, err := NewLearner(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := learner.Fit(dm); err != nil {
		t.Fatal(err)
	}
	margins := make([]float64, dm.NumRow())
	learner.predictMarginsInternal(dm, margins, false)
	for _, m := range margins {
		if math.IsNaN(m) {
			t.Fatal("nan margin")
		}
	}
}

func TestSurvivalAFTTrainSmoke(t *testing.T) {
	vals := []float64{
		1, 0, 0, 1, 1, 1, 0, 0,
		1, 0, 0, 1, 1, 1, 0, 0,
	}
	labels := []float64{1.5, -2, 3, -1, 2, -4, 1, -0.5}
	dm, err := data.NewDense(vals, 8, 2, labels, nil)
	if err != nil {
		t.Fatal(err)
	}
	cfg := Config{
		Objective:    ObjectiveSurvivalAFT,
		NumRound:     2,
		MaxDepth:     2,
		LearningRate: 0.1,
		TreeMethod:   TreeMethodExact,
	}
	learner, err := NewLearner(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := learner.Fit(dm); err != nil {
		t.Fatal(err)
	}
}

func TestSurvivalAFTIntervalTrainSmoke(t *testing.T) {
	vals := []float64{1, 0, 0, 1, 1, 1, 0, 0, 1, 1}
	labels := []float64{1, 1, 1, 1, 1} // placeholder for Dense
	d, err := data.NewDense(vals, 5, 2, labels, nil)
	if err != nil {
		t.Fatal(err)
	}
	ivs := []data.AFTInterval{
		{1.5, 1.5},
		{2, math.Inf(1)},
		{0, 4},
		{1, 6},
		{3, 3},
	}
	dm, err := data.NewAFTDense(d, ivs)
	if err != nil {
		t.Fatal(err)
	}
	cfg := Config{
		Objective:    ObjectiveSurvivalAFT,
		NumRound:     3,
		MaxDepth:     2,
		LearningRate: 0.1,
		TreeMethod:   TreeMethodExact,
	}
	learner, err := NewLearner(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := learner.Fit(dm); err != nil {
		t.Fatal(err)
	}
	margins := make([]float64, dm.NumRow())
	learner.predictMarginsInternal(dm, margins, false)
	for _, m := range margins {
		if math.IsNaN(m) {
			t.Fatal("nan margin")
		}
	}
}
