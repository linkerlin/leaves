package train_test

import (
	"math"
	"path/filepath"
	"testing"

	_ "github.com/dmitryikh/leaves"
	"github.com/dmitryikh/leaves/data"
	"github.com/dmitryikh/leaves/io"
	"github.com/dmitryikh/leaves/metrics"
	"github.com/dmitryikh/leaves/model"
	"github.com/dmitryikh/leaves/train"
	"github.com/dmitryikh/leaves/tree"
)

func syntheticRankingMatrix(t *testing.T) *data.DenseWithGroups {
	t.Helper()
	vals := []float64{
		1.0, 0.0,
		0.5, 0.1,
		0.0, 1.0,
		0.9, 0.0,
		0.4, 0.2,
		0.1, 0.8,
	}
	labels := []float64{3, 1, 0, 2, 1, 0}
	dense, err := data.NewDense(vals, 6, 2, labels, nil)
	if err != nil {
		t.Fatal(err)
	}
	dm, err := data.NewDenseWithGroups(dense, []int{3, 3})
	if err != nil {
		t.Fatal(err)
	}
	return dm
}

func ndcgScore(t *testing.T, dm data.Matrix, preds []float64) float64 {
	t.Helper()
	m, err := metrics.Resolve("ndcg", metrics.Options{Groups: groupsOf(dm)})
	if err != nil {
		t.Fatal(err)
	}
	v, err := m.Evaluate(dm.Labels(), preds)
	if err != nil {
		t.Fatal(err)
	}
	return v
}

func groupsOf(dm data.Matrix) []int {
	if gm, ok := dm.(data.GroupedMatrix); ok {
		return gm.Groups()
	}
	return nil
}

func TestFitRankingNDCGImproves(t *testing.T) {
	dm := syntheticRankingMatrix(t)
	baseline := make([]float64, dm.NumRow())
	before := ndcgScore(t, dm, baseline)

	learner, err := train.NewLearner(train.Config{
		Objective:    train.ObjectiveRankNDCG,
		NumRound:     30,
		MaxDepth:     3,
		LearningRate: 0.3,
		Lambda:       1.0,
		TreeMethod:   train.TreeMethodExact,
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := learner.Fit(dm); err != nil {
		t.Fatal(err)
	}
	if err := train.ValidateRankingData(dm); err != nil {
		t.Fatal(err)
	}

	preds := make([]float64, dm.NumRow())
	if err := learner.PredictMargins(dm, preds); err != nil {
		t.Fatal(err)
	}
	after := ndcgScore(t, dm, preds)
	if after < before-0.01 {
		t.Errorf("NDCG should not drop: before=%f after=%f", before, after)
	}
	if preds[0] <= preds[2] {
		t.Errorf("q1: doc0 (rel=3) margin %f should beat doc2 (rel=0) %f", preds[0], preds[2])
	}
	if preds[3] <= preds[5] {
		t.Errorf("q2: doc0 margin %f should beat doc2 %f", preds[3], preds[5])
	}
}

func TestFitRankingPairwise(t *testing.T) {
	dm := syntheticRankingMatrix(t)
	baseline := make([]float64, dm.NumRow())
	before := ndcgScore(t, dm, baseline)

	learner, err := train.FitRanking(train.Config{
		Objective:    train.ObjectiveRankPairwise,
		NumRound:     30,
		MaxDepth:     3,
		LearningRate: 0.3,
		Lambda:       1.0,
		TreeMethod:   train.TreeMethodExact,
		EvalMetric:   train.EvalNDCG,
	}, dm)
	if err != nil {
		t.Fatal(err)
	}
	if learner.Model() == nil {
		t.Fatal("nil model")
	}

	preds := make([]float64, dm.NumRow())
	if err := learner.PredictMargins(dm, preds); err != nil {
		t.Fatal(err)
	}
	after := ndcgScore(t, dm, preds)
	if after < before-0.01 {
		t.Errorf("NDCG should not drop: before=%f after=%f", before, after)
	}
	if preds[0] <= preds[2] {
		t.Errorf("q1: doc0 (rel=3) margin %f should beat doc2 (rel=0) %f", preds[0], preds[2])
	}
	if preds[3] <= preds[5] {
		t.Errorf("q2: doc0 (rel=2) margin %f should beat doc2 (rel=0) %f", preds[3], preds[5])
	}
}

func TestFitRankingListwise(t *testing.T) {
	dm := syntheticRankingMatrix(t)
	before := ndcgScore(t, dm, make([]float64, dm.NumRow()))

	learner, err := train.NewLearner(train.Config{
		Objective:    train.ObjectiveRankListwise,
		NumRound:     30,
		MaxDepth:     3,
		LearningRate: 0.3,
		Lambda:       1.0,
		TreeMethod:   train.TreeMethodExact,
		EvalMetric:   train.EvalNDCG,
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := learner.Fit(dm); err != nil {
		t.Fatal(err)
	}

	preds := make([]float64, dm.NumRow())
	if err := learner.PredictMargins(dm, preds); err != nil {
		t.Fatal(err)
	}
	after := ndcgScore(t, dm, preds)
	if after < before-0.01 {
		t.Errorf("NDCG should not drop: before=%f after=%f", before, after)
	}
	if preds[0] <= preds[2] {
		t.Errorf("high-rel doc margin %f should beat low-rel %f", preds[0], preds[2])
	}
}

func TestFitRankingRequiresGroups(t *testing.T) {
	vals := []float64{0, 1, 2, 3}
	labels := []float64{1, 0, 2, 0}
	dm, err := data.NewDense(vals, 4, 1, labels, nil)
	if err != nil {
		t.Fatal(err)
	}
	learner, err := train.NewLearner(train.Config{Objective: train.ObjectiveRankNDCG, NumRound: 5})
	if err != nil {
		t.Fatal(err)
	}
	if err := learner.Fit(dm); err == nil {
		t.Fatal("expected error without groups")
	}
}

func TestFitRankingSaveRoundTrip(t *testing.T) {
	dm := syntheticRankingMatrix(t)
	learner, err := train.NewLearner(train.Config{
		Objective:    train.ObjectiveRankNDCG,
		NumRound:     15,
		MaxDepth:     2,
		LearningRate: 0.4,
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := learner.Fit(dm); err != nil {
		t.Fatal(err)
	}

	mem := make([]float64, dm.NumRow())
	learner.PredictMargins(dm, mem)

	path := filepath.Join(t.TempDir(), "rank.leaves.json")
	if err := learner.Save(path); err != nil {
		t.Fatal(err)
	}
	loaded, err := io.LoadLeavesJSONFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if loaded.Objective != train.ObjectiveRankNDCG {
		t.Errorf("objective: got %q", loaded.Objective)
	}
	eng, err := model.NewEngine(loaded.IR, tree.ApplyTransformRaw, tree.TransformRaw, tree.BackendNative)
	if err != nil {
		t.Fatal(err)
	}
	filePreds := make([]float64, dm.NumRow())
	for i := 0; i < dm.NumRow(); i++ {
		row := make([]float64, 2)
		_ = dm.Row(i, row)
		_ = eng.Predict(row, 0, filePreds[i:i+1])
	}
	for i := range mem {
		if math.Abs(mem[i]-filePreds[i]) > 1e-5 {
			t.Errorf("sample %d: mem %f file %f", i, mem[i], filePreds[i])
		}
	}
}
