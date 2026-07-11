package train_test

import (
	"path/filepath"
	"testing"

	"github.com/linkerlin/leaves/data"
	leavesio "github.com/linkerlin/leaves/io"
	"github.com/linkerlin/leaves/train"
)

// TestMultiOutputTreeTrains 向量叶 multi_output_tree 端到端：
// 多目标回归开 MultiOutputTree=true，应得到 OutputDim=numTarget 的向量叶树，
// 训练收敛（rmse 下降），且 leaves.json 往返保留 OutputDim。
func TestMultiOutputTreeTrains(t *testing.T) {
	const n, p, k = 60, 2, 2
	vals := make([]float64, n*p)
	targets := make([]float64, n*k)
	for i := 0; i < n; i++ {
		x0 := float64(i%10) * 0.1
		x1 := float64(i%7) * 0.15
		vals[i*p+0] = x0
		vals[i*p+1] = x1
		targets[i*k+0] = x0 + 0.1*x1
		targets[i*k+1] = 2*x1 - 0.05*x0
	}
	dm, err := data.NewMultiTargetDense(vals, n, p, targets, k, nil)
	if err != nil {
		t.Fatal(err)
	}

	learner, err := train.NewLearner(train.Config{
		Objective:       "reg:squarederror",
		EvalMetric:      "rmse",
		NumTarget:       2,
		MultiOutputTree: true,
		NumRound:        40,
		MaxDepth:        3,
		LearningRate:    0.25,
		Lambda:          1,
		Seed:            7,
		TreeMethod:      "hist",
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := learner.Fit(dm); err != nil {
		t.Fatal(err)
	}

	// 结构：每棵树 OutputDim == numTarget（向量叶），非 one_output_per_tree 的标量树。
	ir := learner.Model()
	if ir == nil || ir.Forest == nil || len(ir.Forest.Trees) == 0 {
		t.Fatal("nil/empty model")
	}
	if ir.Forest.NumOutputGroups != 2 {
		t.Fatalf("NumOutputGroups=%d want 2", ir.Forest.NumOutputGroups)
	}
	for ti, tr := range ir.Forest.Trees {
		if tr.OutputDim != 2 {
			t.Fatalf("tree %d OutputDim=%d want 2 (vector leaf)", ti, tr.OutputDim)
		}
		if len(tr.LeafValue)%2 != 0 {
			t.Fatalf("tree %d LeafValue len=%d not multiple of OutputDim=2", ti, len(tr.LeafValue))
		}
	}

	// 收敛：train rmse 应较低（向量叶共享分裂对相关目标仍能学）。
	score, err := learner.Eval(dm)
	if err != nil {
		t.Fatal(err)
	}
	if score > 0.35 {
		t.Fatalf("train rmse too high (did not learn): %g", score)
	}

	// leaves.json 往返：OutputDim 保留。
	path := filepath.Join(t.TempDir(), "mot.leaves.json")
	if err := learner.Save(path); err != nil {
		t.Fatal(err)
	}
	loaded, _, err := leavesio.ParseLeavesJSONFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if loaded == nil || loaded.Forest == nil || len(loaded.Forest.Trees) == 0 {
		t.Fatal("loaded nil/empty")
	}
	for ti, tr := range loaded.Forest.Trees {
		if tr.OutputDim != 2 {
			t.Fatalf("loaded tree %d OutputDim=%d want 2 (round-trip lost OutputDim)", ti, tr.OutputDim)
		}
	}
}

// TestMultiOutputTreeRejectsSingleOutput MultiOutputTree 需多输出（g>1）；单输出应在 NewLearner 被拒。
func TestMultiOutputTreeRejectsSingleOutput(t *testing.T) {
	_, err := train.NewLearner(train.Config{
		Objective:       "binary:logistic",
		MultiOutputTree: true,
		NumRound:        3,
		MaxDepth:        2,
		Seed:            1,
	})
	if err == nil {
		t.Fatal("expected error at NewLearner: MultiOutputTree requires multi-output (numClass>1 or NumTarget>1)")
	}
}
