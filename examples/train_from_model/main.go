// 从已有 XGBoost JSON 推断 objective，嗅探 TSV 训练数据并训练新模型。
//
//	go run ./examples/train_from_model/ -model testdata/xgboost_smoke.json -data testdata/breast_cancer_train.tsv
package main

import (
	"flag"
	"fmt"
	"log"
	"os"

	"github.com/dmitryikh/leaves"
	"github.com/dmitryikh/leaves/data"
)

func main() {
	modelPath := flag.String("model", "testdata/xgboost_smoke.json", "XGBoost/leaves JSON（仅用于推断 objective）")
	dataPath := flag.String("data", "testdata/breast_cancer_train.tsv", "训练数据（自动嗅探格式）")
	rounds := flag.Int("rounds", 10, "提升轮数")
	flag.Parse()

	if _, err := os.Stat(*dataPath); err != nil {
		log.Fatalf("data file: %v", err)
	}

	obj, err := leaves.InferObjectiveFromModel(*modelPath)
	if err != nil {
		log.Fatalf("infer objective: %v", err)
	}
	fmt.Printf("inferred objective: %s\n", obj)

	learner, err := leaves.NewLearnerFromModelAndData(*modelPath, *dataPath, leaves.TrainConfig{
		NumRound:     *rounds,
		MaxDepth:     4,
		LearningRate: 0.1,
		TreeMethod:   leaves.TrainTreeMethodExact,
	}, data.DefaultFileLoadOptions())
	if err != nil {
		log.Fatalf("train: %v", err)
	}

	out := "train_from_model.leaves.json"
	if err := learner.Save(out); err != nil {
		log.Fatalf("save: %v", err)
	}
	fmt.Printf("saved %s (%d trees)\n", out, len(learner.Model().Forest.Trees))
}
