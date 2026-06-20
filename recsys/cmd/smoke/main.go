// 端到端推荐 smoke：数据准备 → 召回(100/User) → LTR 排序 → 发牌。
package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/linkerlin/leaves/recsys"
	"github.com/linkerlin/leaves/recsys/pipeline"
)

func main() {
	ws := flag.String("workspace", "recsys/out/smoke", "工作区根目录")
	seed := flag.Int64("seed", 42, "随机种子")
	trainUsers := flag.Int("train-users", 18, "训练用户数")
	testUsers := flag.Int("test-users", 6, "测试用户数")
	recallSize := flag.Int("recall-size", 100, "每用户召回 Item 数")
	flag.Parse()

	cfg := recsys.DefaultSmokeConfig()
	cfg.Seed = *seed
	cfg.TrainUsers = *trainUsers
	cfg.TestUsers = *testUsers
	cfg.RecallSize = *recallSize
	cfg.NumItems = 512
	if cfg.NumItems < cfg.RecallSize {
		cfg.NumItems = cfg.RecallSize * 4
	}

	w, err := recsys.NewWorkspace(*ws)
	if err != nil {
		fatal(err)
	}

	res, err := pipeline.Run(w, cfg)
	if err != nil {
		fatal(err)
	}

	fmt.Printf("=== recsys smoke 完成 ===\n")
	fmt.Printf("工作区: %s\n", w.Root)
	fmt.Printf("用户: train=%d test=%d\n", res.Prep.TrainUsers, res.Prep.TestUsers)
	fmt.Printf("召回: train=%d test=%d 行 (每用户 %d Item)\n",
		res.RecallTrain, res.RecallTest, cfg.RecallSize)
	fmt.Printf("排序 TSV: train=%d test=%d 行\n", res.RankTrain, res.RankTest)
	fmt.Printf("NDCG@%d: train=%.4f test=%.4f\n", res.Eval.NDCGK, res.Eval.TrainNDCG, res.Eval.TestNDCG)
	fmt.Printf("发牌: %d 行 → %s\n", res.DealRows, w.DealTest())
	fmt.Printf("模型: %s\n", w.ModelPath())
}

func fatal(err error) {
	fmt.Fprintf(os.Stderr, "错误: %v\n", err)
	os.Exit(1)
}
