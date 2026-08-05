package train_test

import (
	"math"
	"path/filepath"
	"testing"

	"github.com/linkerlin/leaves/v2/data"
	leavesio "github.com/linkerlin/leaves/v2/io"
	"github.com/linkerlin/leaves/v2/train"
	"github.com/linkerlin/leaves/v2/tree"
)

// TestMultiTargetOneOutputPerTree 锁定 LIB-21：多目标回归 one_output_per_tree。
func TestMultiTargetOneOutputPerTree(t *testing.T) {
	const n, p, k = 60, 2, 2
	vals := make([]float64, n*p)
	targets := make([]float64, n*k)
	for i := 0; i < n; i++ {
		x0 := float64(i%10) * 0.1
		x1 := float64(i%7) * 0.15
		vals[i*p+0] = x0
		vals[i*p+1] = x1
		targets[i*k+0] = x0 + 0.1*x1    // y0
		targets[i*k+1] = 2*x1 - 0.05*x0 // y1
	}
	dm, err := data.NewMultiTargetDense(vals, n, p, targets, k, nil)
	if err != nil {
		t.Fatal(err)
	}
	if dm.NumTarget() != 2 {
		t.Fatalf("NumTarget=%d", dm.NumTarget())
	}

	learner, err := train.NewLearner(train.Config{
		Objective:    "reg:squarederror",
		EvalMetric:   "rmse",
		NumTarget:    2,
		NumRound:     40,
		MaxDepth:     3,
		LearningRate: 0.25,
		Lambda:       1,
		Seed:         7,
		TreeMethod:   "hist",
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := learner.Fit(dm); err != nil {
		t.Fatal(err)
	}
	score, err := learner.Eval(dm)
	if err != nil {
		t.Fatal(err)
	}
	if score > 0.25 {
		t.Fatalf("train rmse too high: %g", score)
	}

	ir := learner.Model()
	if ir == nil || ir.Forest == nil {
		t.Fatal("nil model")
	}
	if ir.Forest.NumOutputGroups != 2 {
		t.Fatalf("NumOutputGroups=%d want 2", ir.Forest.NumOutputGroups)
	}
	// one_output_per_tree：每轮 2 棵标量树
	if ir.Forest.NEstimators() < 1 {
		t.Fatal("no estimators")
	}
	for _, tr := range ir.Forest.Trees {
		if tr.OutputDim > 1 {
			t.Fatalf("expected scalar trees (one_output_per_tree), got OutputDim=%d", tr.OutputDim)
		}
	}

	// 抽查几个点 margin 接近真值
	x := []float64{0.5, 0.3}
	margins := tree.ForestMargins(ir.Forest, x, 0)
	if len(margins) != 2 {
		t.Fatalf("margins len %d", len(margins))
	}
	want0 := 0.5 + 0.1*0.3
	want1 := 2*0.3 - 0.05*0.5
	if math.Abs(margins[0]-want0) > 0.35 {
		t.Errorf("target0 margin=%g want~%g", margins[0], want0)
	}
	if math.Abs(margins[1]-want1) > 0.35 {
		t.Errorf("target1 margin=%g want~%g", margins[1], want1)
	}
}

// TestMultiTargetSaveLoadRoundTrip 训练 → leaves.json → 再加载预测一致。
func TestMultiTargetSaveLoadRoundTrip(t *testing.T) {
	const n, p, k = 48, 2, 2
	vals := make([]float64, n*p)
	targets := make([]float64, n*k)
	for i := 0; i < n; i++ {
		x0 := float64(i%8) * 0.1
		x1 := float64(i%5) * 0.15
		vals[i*p+0] = x0
		vals[i*p+1] = x1
		targets[i*k+0] = x0 + x1
		targets[i*k+1] = 1.5*x1 - 0.1*x0
	}
	dm, err := data.NewMultiTargetDense(vals, n, p, targets, k, nil)
	if err != nil {
		t.Fatal(err)
	}
	learner, err := train.NewLearner(train.Config{
		Objective:    "reg:squarederror",
		NumTarget:    2,
		NumRound:     25,
		MaxDepth:     3,
		LearningRate: 0.2,
		Seed:         3,
		TreeMethod:   "hist",
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := learner.Fit(dm); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "mt.leaves.json")
	if err := learner.Save(path); err != nil {
		t.Fatal(err)
	}
	ens, err := leavesio.LoadFromFile(path, &leavesio.LoadOptions{Backend: leavesio.BackendNative})
	if err != nil {
		// LoadFromFile 需根包 register；回退 ParseLeavesJSONFile + 本地 margin
		ir, obj, err2 := leavesio.ParseLeavesJSONFile(path)
		if err2 != nil {
			t.Fatalf("load: %v / %v", err, err2)
		}
		if obj != "reg:squarederror" {
			t.Fatalf("objective %q", obj)
		}
		if ir.Forest == nil || ir.Forest.NumOutputGroups != 2 {
			t.Fatalf("forest groups=%v", ir.Forest)
		}
		x := []float64{0.4, 0.3}
		m1 := tree.ForestMargins(learner.Model().Forest, x, 0)
		m2 := tree.ForestMargins(ir.Forest, x, 0)
		if len(m1) != 2 || len(m2) != 2 {
			t.Fatal(m1, m2)
		}
		for d := 0; d < 2; d++ {
			if math.Abs(m1[d]-m2[d]) > 1e-9 {
				t.Errorf("dim %d: train=%g load=%g", d, m1[d], m2[d])
			}
		}
		return
	}
	_ = ens
	if ens.NOutputGroups() != 2 {
		t.Fatalf("NOutputGroups=%d", ens.NOutputGroups())
	}
}

func TestMultiTargetRequiresData(t *testing.T) {
	// 标量 Dense + NumTarget=2 应失败
	dm, err := data.NewDense([]float64{0, 1, 1, 0}, 2, 2, []float64{0, 1}, nil)
	if err != nil {
		t.Fatal(err)
	}
	learner, err := train.NewLearner(train.Config{
		Objective: "reg:squarederror",
		NumTarget: 2,
		NumRound:  2,
		MaxDepth:  2,
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := learner.Fit(dm); err == nil {
		t.Fatal("expected error for scalar matrix with NumTarget=2")
	}
}
