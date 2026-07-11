// 演示如何通过 objective.Register / metrics.Register 扩展训练目标与指标，
// 无需改 leaves 内核 switch。详见 docs/extension-points.md。
//
//	go run ./examples/extension/
//	go test ./examples/extension/ -count=1
package main

import (
	"fmt"
	"log"
	"math"
	"os"
	"path/filepath"

	"github.com/linkerlin/leaves/data"
	"github.com/linkerlin/leaves/metrics"
	"github.com/linkerlin/leaves/objective"
	"github.com/linkerlin/leaves/train"
)

func init() {
	// 自定义目标：绝对损失（L1 / MAE 风格），hess 取常数以利树分裂。
	objective.Register("custom:l1", func(int) (objective.Func, error) {
		return L1{}, nil
	})
	// 自定义指标：最大绝对误差（越低越好）。
	metrics.Register("max_abs_error", func(metrics.Options) (metrics.Metric, error) {
		return MaxAbsError{}, nil
	})
}

// L1 对标 MAE 风格目标（非内置；仅本示例注册）。
type L1 struct{}

func (L1) Name() string { return "custom:l1" }

func (L1) GradHess(pred, label, weight float64) (grad, hess float64) {
	if weight <= 0 {
		weight = 1
	}
	d := pred - label
	switch {
	case d > 0:
		grad = weight
	case d < 0:
		grad = -weight
	default:
		grad = 0
	}
	// 恒定 Hessian：避免零二阶导致无法分裂（常见 L1 树技巧）。
	hess = weight
	return grad, hess
}

func (L1) InitialPred(labels []float64, weights []float64) float64 {
	if len(labels) == 0 {
		return 0
	}
	// 中位数初始化更贴 L1；小样本用均值即可。
	var sw, sy float64
	for i, y := range labels {
		w := 1.0
		if weights != nil && i < len(weights) {
			w = weights[i]
		}
		sw += w
		sy += w * y
	}
	if sw == 0 {
		return 0
	}
	return sy / sw
}

// MaxAbsError max_i |y_i - ŷ_i|。
type MaxAbsError struct{}

func (MaxAbsError) Name() string         { return "max_abs_error" }
func (MaxAbsError) HigherIsBetter() bool { return false }

func (MaxAbsError) Evaluate(yTrue, yPred []float64) (float64, error) {
	if len(yTrue) != len(yPred) || len(yTrue) == 0 {
		return 0, fmt.Errorf("max_abs_error: length mismatch or empty")
	}
	m := 0.0
	for i := range yTrue {
		a := math.Abs(yTrue[i] - yPred[i])
		if a > m {
			m = a
		}
	}
	return m, nil
}

func (MaxAbsError) EvaluatePerGroup(yTrue, yPred []float64, _ []int) (float64, error) {
	return MaxAbsError{}.Evaluate(yTrue, yPred)
}

func main() {
	// 合成数据：y ≈ 2*x0 + x1 + 噪声
	const n, p = 80, 2
	vals := make([]float64, n*p)
	labels := make([]float64, n)
	for i := 0; i < n; i++ {
		x0 := float64(i%10) / 10
		x1 := float64(i%7) / 7
		vals[i*p+0] = x0
		vals[i*p+1] = x1
		labels[i] = 2*x0 + x1 + 0.05*float64(i%3)
	}
	dm, err := data.NewDense(vals, n, p, labels, nil)
	if err != nil {
		log.Fatalf("dense: %v", err)
	}

	learner, err := train.NewLearner(train.Config{
		Objective:    "custom:l1",
		EvalMetric:   "max_abs_error",
		NumRound:     30,
		MaxDepth:     3,
		LearningRate: 0.2,
		Lambda:       1,
		Seed:         42,
		TreeMethod:   "hist",
	})
	if err != nil {
		log.Fatalf("NewLearner: %v", err)
	}
	if err := learner.Fit(dm); err != nil {
		log.Fatalf("Fit: %v", err)
	}
	score, err := learner.Eval(dm)
	if err != nil {
		log.Fatalf("Eval: %v", err)
	}
	fmt.Printf("custom:l1 + max_abs_error → train max_abs_error=%.6f rounds=%d\n",
		score, learner.BoostRounds())

	out := filepath.Join(os.TempDir(), "extension_demo.leaves.json")
	if err := learner.Save(out); err != nil {
		log.Fatalf("Save: %v", err)
	}
	fmt.Printf("saved %s\n", out)
	fmt.Println("ok: extension via Register only (no kernel switch)")
}
