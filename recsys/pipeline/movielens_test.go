package pipeline_test

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/linkerlin/leaves/v2/data"
	"github.com/linkerlin/leaves/v2/recsys"
	"github.com/linkerlin/leaves/v2/recsys/movielens"
	"github.com/linkerlin/leaves/v2/recsys/pipeline"
	"github.com/linkerlin/leaves/v2/recsys/recall"
	"github.com/linkerlin/leaves/v2/recsys/tsvio"
)

func TestMovieLensFourStage(t *testing.T) {
	if testing.Short() {
		t.Skip("short")
	}
	root := findRepoRoot(t)
	mlCfg := movielens.DefaultConfig()
	mlCfg.RepoRoot = root
	// 缩小规模以加速 CI / 本地回归
	mlCfg.TrainUsers = 20
	mlCfg.TestUsers = 5

	ds, titles, err := movielens.Load(mlCfg)
	if err != nil {
		t.Skipf("MovieLens load (need network or .cache/ml-100k.zip): %v", err)
	}
	if len(titles) == 0 {
		t.Fatal("empty titles")
	}
	if len(ds.Catalog) < 100 {
		t.Fatalf("catalog too small: %d", len(ds.Catalog))
	}

	dir := t.TempDir()
	w, err := recsys.NewWorkspace(dir)
	if err != nil {
		t.Fatal(err)
	}
	if err := movielens.WriteTitles(filepath.Join(w.MetaDir(), "item_titles.tsv"), titles); err != nil {
		t.Fatal(err)
	}

	cfg := recsys.DefaultSmokeConfig()
	cfg.Seed = mlCfg.Seed
	cfg.TrainUsers = mlCfg.TrainUsers
	cfg.TestUsers = mlCfg.TestUsers
	cfg.RecallSize = 100
	cfg.TrainRounds = 12
	cfg.DeckSize = 10
	cfg.MaxSameTag = 3
	cfg.NumItems = len(ds.Catalog)

	res, err := pipeline.RunFromDataset(w, ds, cfg)
	if err != nil {
		t.Fatal(err)
	}

	wantTrain := cfg.TrainUsers * cfg.RecallSize
	wantTest := cfg.TestUsers * cfg.RecallSize
	if res.RecallTrain != wantTrain || res.RecallTest != wantTest {
		t.Fatalf("recall rows train=%d test=%d want %d/%d",
			res.RecallTrain, res.RecallTest, wantTrain, wantTest)
	}
	_, recallTest, err := tsvio.ReadRecall(w.RecallTest())
	if err != nil {
		t.Fatal(err)
	}
	if err := recall.Validate(recallTest, 100); err != nil {
		t.Fatal(err)
	}
	dm, err := data.LoadRankingTSV(w.RankTest(), "\t")
	if err != nil {
		t.Fatal(err)
	}
	if dm.NumRow() != wantTest {
		t.Fatalf("rank test rows %d want %d", dm.NumRow(), wantTest)
	}
	if res.Eval.TestNDCG <= 0 {
		t.Fatalf("expected positive test NDCG, got %f", res.Eval.TestNDCG)
	}
	if res.DealRows == 0 || res.DealRows > cfg.TestUsers*cfg.DeckSize {
		t.Fatalf("deal rows %d unexpected", res.DealRows)
	}
	if _, err := os.Stat(w.ModelPath()); err != nil {
		t.Fatal(err)
	}
	// 发牌行 Item 应能映射到片名
	dealPath := w.DealTest()
	b, err := os.ReadFile(dealPath)
	if err != nil {
		t.Fatal(err)
	}
	if len(b) < 20 {
		t.Fatal("deal file too small")
	}
	loaded, err := movielens.LoadTitles(filepath.Join(w.MetaDir(), "item_titles.tsv"))
	if err != nil || len(loaded) == 0 {
		t.Fatalf("titles: %v len=%d", err, len(loaded))
	}
}

// TestMovieLensTimeSplitFourStage 真实数据（u.data 真实时间戳）上的时间切分四段验证。
func TestMovieLensTimeSplitFourStage(t *testing.T) {
	if testing.Short() {
		t.Skip("short")
	}
	root := findRepoRoot(t)
	mlCfg := movielens.DefaultConfig()
	mlCfg.RepoRoot = root
	mlCfg.TrainUsers = 20
	mlCfg.TestUsers = 5

	ds, _, err := movielens.Load(mlCfg)
	if err != nil {
		t.Skipf("MovieLens load (need network or .cache/ml-100k.zip): %v", err)
	}
	for i, r := range ds.Raw {
		if r.Time.IsZero() {
			t.Fatalf("interaction %d missing real timestamp from u.data", i)
		}
	}

	dir := t.TempDir()
	w, err := recsys.NewWorkspace(dir)
	if err != nil {
		t.Fatal(err)
	}
	cfg := recsys.DefaultSmokeConfig()
	cfg.Seed = mlCfg.Seed
	cfg.RecallSize = 100
	cfg.TrainRounds = 12
	cfg.DeckSize = 10
	cfg.MaxSameTag = 3
	cfg.NumItems = len(ds.Catalog)
	cfg.SplitMode = "time"

	res, err := pipeline.RunFromDataset(w, ds, cfg)
	if err != nil {
		t.Fatal(err)
	}
	if res.Prep.SplitMode != "time" {
		t.Fatalf("split mode: got %q want time", res.Prep.SplitMode)
	}
	if res.Prep.TrainRows == 0 || res.Prep.TestRows == 0 {
		t.Fatalf("empty split: train=%d test=%d", res.Prep.TrainRows, res.Prep.TestRows)
	}
	// 行数对账：train+test+重复丢弃+隔离带+val == 全部原始行
	dup := res.Prep.Dropped["duplicate_user_item_train"] + res.Prep.Dropped["duplicate_user_item_test"]
	used := res.Prep.TrainRows + res.Prep.TestRows + dup +
		res.Prep.Dropped["time_isolated"] + res.Prep.Dropped["time_val_unused"]
	if used != len(ds.Raw) {
		t.Fatalf("row accounting: used=%d want %d", used, len(ds.Raw))
	}
	if res.Eval.TestNDCG <= 0 {
		t.Fatalf("expected positive test NDCG, got %f", res.Eval.TestNDCG)
	}
	if res.DealRows == 0 || res.DealRows > res.Prep.TestUsers*cfg.DeckSize {
		t.Fatalf("deal rows %d unexpected (test users %d)", res.DealRows, res.Prep.TestUsers)
	}
}

func findRepoRoot(t *testing.T) string {
	t.Helper()
	cwd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
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
	t.Fatal("go.mod not found")
	return ""
}
