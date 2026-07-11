// 多目标回归 one_output_per_tree 示例（LIB-21）。
//
//	go run ./examples/multitarget/
//	go test ./examples/multitarget/ -count=1
package main

import (
	"fmt"
	"log"
	"os"
	"path/filepath"

	"github.com/linkerlin/leaves/data"
	"github.com/linkerlin/leaves/train"
	"github.com/linkerlin/leaves/tree"
)

func main() {
	const n, p, k = 80, 2, 2
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
		log.Fatal(err)
	}
	learner, err := train.NewLearner(train.Config{
		Objective:    "reg:squarederror",
		EvalMetric:   "rmse",
		NumTarget:    2,
		NumRound:     40,
		MaxDepth:     3,
		LearningRate: 0.25,
		Seed:         42,
		TreeMethod:   "hist",
	})
	if err != nil {
		log.Fatal(err)
	}
	if err := learner.Fit(dm); err != nil {
		log.Fatal(err)
	}
	score, err := learner.Eval(dm)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("multi-target rmse=%.6f groups=%d\n", score, learner.Model().Forest.NumOutputGroups)

	x := []float64{0.5, 0.3}
	m := tree.ForestMargins(learner.Model().Forest, x, 0)
	fmt.Printf("x=%v margins=%v (want ~ [%.3f, %.3f])\n",
		x, m, 0.5+0.1*0.3, 2*0.3-0.05*0.5)

	out := filepath.Join(os.TempDir(), "multitarget_demo.leaves.json")
	if err := learner.Save(out); err != nil {
		log.Fatal(err)
	}
	fmt.Printf("saved %s\n", out)
	fmt.Println("CLI: leaves train --data mt.csv --objective reg:squarederror --num-target 2")
}
