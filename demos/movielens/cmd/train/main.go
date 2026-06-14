// 训练 MovieLens 100K 排序模型并保存 leaves.json。
package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"

	"github.com/dmitryikh/leaves/data"
	"github.com/dmitryikh/leaves/demos/movielens/rankutil"
	"github.com/dmitryikh/leaves/io"
	"github.com/dmitryikh/leaves/train"
)

const (
	defaultRounds   = 40
	defaultDepth    = 4
	defaultLR       = 0.1
	defaultLambda   = 1.0
	defaultNDCGK    = 10
	defaultSeed     = 42
)

func main() {
	objective := flag.String("objective", train.ObjectiveRankNDCG, "rank:ndcg | rank:pairwise | rank:listwise")
	outPath := flag.String("out", "", "输出 model.leaves.json（默认 demos/movielens/out/）")
	flag.Parse()

	dataDir, err := rankutil.DataDir()
	if err != nil {
		fatal(err)
	}
	trainPath := filepath.Join(dataDir, "rank_movielens_train.tsv")
	testPath := filepath.Join(dataDir, "rank_movielens_test.tsv")

	trainDM, err := data.LoadRankingTSV(trainPath, "\t")
	if err != nil {
		fatal(fmt.Errorf("load train: %w", err))
	}
	testDM, err := data.LoadRankingTSV(testPath, "\t")
	if err != nil {
		fatal(fmt.Errorf("load test: %w", err))
	}

	cfg := train.Config{
		Objective:    *objective,
		NumRound:     defaultRounds,
		MaxDepth:     defaultDepth,
		LearningRate: defaultLR,
		Lambda:       defaultLambda,
		TreeMethod:   train.TreeMethodHist,
		Seed:         defaultSeed,
		NDCGK:        defaultNDCGK,
		EvalMetric:   fmt.Sprintf("ndcg@%d", defaultNDCGK),
	}
	learner, err := train.NewLearner(cfg)
	if err != nil {
		fatal(err)
	}
	fmt.Printf("训练 %s：%d 轮，%d 用户，%d 特征 …\n",
		*objective, cfg.NumRound, len(trainDM.Groups()), trainDM.Cols)

	if err := learner.Fit(trainDM); err != nil {
		fatal(err)
	}

	trainPred, err := rankutil.PredictMargins(learner, trainDM)
	if err != nil {
		fatal(err)
	}
	testPred, err := rankutil.PredictMargins(learner, testDM)
	if err != nil {
		fatal(err)
	}
	trainNDCG, err := rankutil.NDCGAtK(trainDM, trainPred, defaultNDCGK)
	if err != nil {
		fatal(err)
	}
	testNDCG, err := rankutil.NDCGAtK(testDM, testPred, defaultNDCGK)
	if err != nil {
		fatal(err)
	}
	fmt.Printf("leaves  NDCG@%d  train=%.4f  test=%.4f\n", defaultNDCGK, trainNDCG, testNDCG)

	blPath := rankutil.BaselinePath(dataDir, *objective)
	if bl, err := rankutil.LoadXGBBaseline(blPath); err == nil {
		fmt.Printf("XGBoost NDCG@%d  train=%.4f  test=%.4f  (baseline %s)\n",
			bl.NDCGK, bl.FinalTrainNDCG, bl.FinalTestNDCG, filepath.Base(blPath))
		fmt.Printf("  train Δ=%.4f  test Δ=%.4f (leaves - xgb)\n",
			trainNDCG-bl.FinalTrainNDCG, testNDCG-bl.FinalTestNDCG)
	} else if *objective != train.ObjectiveRankListwise {
		fmt.Fprintf(os.Stderr, "警告: 未找到 baseline %s (%v)\n", blPath, err)
	}

	savePath := *outPath
	if savePath == "" {
		outDir, err := rankutil.OutDir()
		if err != nil {
			fatal(err)
		}
		name := "model_" + sanitizeObjective(*objective) + ".leaves.json"
		savePath = filepath.Join(outDir, name)
	}
	if err := io.SaveTrainModel(savePath, learner.Model(), *objective); err != nil {
		fatal(err)
	}
	fmt.Printf("已保存 %s\n", savePath)
}

func sanitizeObjective(obj string) string {
	out := make([]byte, 0, len(obj))
	for i := 0; i < len(obj); i++ {
		c := obj[i]
		if c == ':' {
			out = append(out, '_')
			continue
		}
		out = append(out, c)
	}
	return string(out)
}

func fatal(err error) {
	fmt.Fprintf(os.Stderr, "错误: %v\n", err)
	os.Exit(1)
}
