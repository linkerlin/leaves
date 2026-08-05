// MovieLens 100K 四段推荐流水线：准备 → 召回(100/User) → LTR 排序 → 发牌。
//
//	go run ./recsys/cmd/movielens
//	go run ./recsys/cmd/movielens -workspace demos/movielens/out/fourstage -recall-size 100
package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"

	"github.com/linkerlin/leaves/v2/recsys"
	"github.com/linkerlin/leaves/v2/recsys/movielens"
	"github.com/linkerlin/leaves/v2/recsys/pipeline"
)

func main() {
	ws := flag.String("workspace", "demos/movielens/out/fourstage", "工作区根目录")
	seed := flag.Int64("seed", 42, "随机种子（用户切分）")
	trainUsers := flag.Int("train-users", 60, "训练用户数")
	testUsers := flag.Int("test-users", 15, "测试用户数")
	recallSize := flag.Int("recall-size", 100, "每用户召回 Item 数")
	rounds := flag.Int("rounds", 30, "LTR 训练轮数")
	deck := flag.Int("deck-size", 10, "发牌 Top-K")
	maxTag := flag.Int("max-same-tag", 3, "同 Tag 上限")
	cache := flag.String("cache", "", "ml-100k.zip 路径（默认 .cache/ml-100k.zip）")
	flag.Parse()

	root := findRepoRoot()
	mlCfg := movielens.DefaultConfig()
	mlCfg.Seed = *seed
	mlCfg.TrainUsers = *trainUsers
	mlCfg.TestUsers = *testUsers
	mlCfg.RepoRoot = root
	if *cache != "" {
		mlCfg.CachePath = *cache
	}

	fmt.Fprintf(os.Stderr, "loading MovieLens 100K …\n")
	ds, titles, err := movielens.Load(mlCfg)
	if err != nil {
		fatal(err)
	}

	cfg := recsys.DefaultSmokeConfig()
	cfg.Seed = *seed
	cfg.TrainUsers = *trainUsers
	cfg.TestUsers = *testUsers
	cfg.RecallSize = *recallSize
	cfg.TrainRounds = *rounds
	cfg.DeckSize = *deck
	cfg.MaxSameTag = *maxTag
	cfg.NumItems = len(ds.Catalog)

	w, err := recsys.NewWorkspace(*ws)
	if err != nil {
		fatal(err)
	}
	if err := movielens.WriteTitles(filepath.Join(w.MetaDir(), "item_titles.tsv"), titles); err != nil {
		fatal(err)
	}

	fmt.Fprintf(os.Stderr, "pipeline: prep→recall→rank→deal (catalog=%d) …\n", len(ds.Catalog))
	res, err := pipeline.RunFromDataset(w, ds, cfg)
	if err != nil {
		fatal(err)
	}

	fmt.Printf("=== MovieLens four-stage 完成 ===\n")
	fmt.Printf("工作区: %s\n", w.Root)
	fmt.Printf("用户: train=%d test=%d\n", res.Prep.TrainUsers, res.Prep.TestUsers)
	fmt.Printf("catalog: %d items\n", res.Prep.CatalogSize)
	fmt.Printf("召回: train=%d test=%d 行 (每用户 %d)\n",
		res.RecallTrain, res.RecallTest, cfg.RecallSize)
	fmt.Printf("排序 TSV: train=%d test=%d 行\n", res.RankTrain, res.RankTest)
	fmt.Printf("NDCG@%d: train=%.4f test=%.4f\n", res.Eval.NDCGK, res.Eval.TrainNDCG, res.Eval.TestNDCG)
	fmt.Printf("发牌: %d 行 → %s\n", res.DealRows, w.DealTest())
	fmt.Printf("模型: %s\n", w.ModelPath())
	fmt.Printf("片名: %s\n", filepath.Join(w.MetaDir(), "item_titles.tsv"))
}

func findRepoRoot() string {
	cwd, _ := os.Getwd()
	dir := cwd
	for i := 0; i < 8; i++ {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir
		}
		p := filepath.Dir(dir)
		if p == dir {
			break
		}
		dir = p
	}
	return cwd
}

func fatal(err error) {
	fmt.Fprintf(os.Stderr, "错误: %v\n", err)
	os.Exit(1)
}
