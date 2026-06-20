package pipeline_test

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/linkerlin/leaves/data"
	"github.com/linkerlin/leaves/recsys"
	"github.com/linkerlin/leaves/recsys/pipeline"
	"github.com/linkerlin/leaves/recsys/recall"
	"github.com/linkerlin/leaves/recsys/tsvio"
)

func TestSmokePipeline100PerUser(t *testing.T) {
	dir := t.TempDir()
	w, err := recsys.NewWorkspace(dir)
	if err != nil {
		t.Fatal(err)
	}
	cfg := recsys.DefaultSmokeConfig()
	cfg.RecallSize = 100
	cfg.NumItems = 512

	res, err := pipeline.Run(w, cfg)
	if err != nil {
		t.Fatal(err)
	}

	wantTrainRows := cfg.TrainUsers * cfg.RecallSize
	wantTestRows := cfg.TestUsers * cfg.RecallSize
	if res.RecallTrain != wantTrainRows {
		t.Fatalf("recall train rows: got %d want %d", res.RecallTrain, wantTrainRows)
	}
	if res.RecallTest != wantTestRows {
		t.Fatalf("recall test rows: got %d want %d", res.RecallTest, wantTestRows)
	}
	if res.RankTrain != wantTrainRows || res.RankTest != wantTestRows {
		t.Fatalf("rank rows mismatch: train=%d test=%d", res.RankTrain, res.RankTest)
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
	if dm.NumRow() != wantTestRows {
		t.Fatalf("rank test tsv rows: got %d want %d", dm.NumRow(), wantTestRows)
	}
	groups := dm.Groups()
	if len(groups) != cfg.TestUsers {
		t.Fatalf("test groups: got %d want %d", len(groups), cfg.TestUsers)
	}
	for i, g := range groups {
		if g != cfg.RecallSize {
			t.Fatalf("group %d size %d want %d", i, g, cfg.RecallSize)
		}
	}

	format := data.DetectFileFormat(w.RankTrain())
	if format != data.FormatRanking {
		t.Fatalf("sniff format: got %v want FormatRanking", format)
	}

	if res.Eval.TestNDCG <= 0 {
		t.Fatalf("expected positive test NDCG, got %f", res.Eval.TestNDCG)
	}
	if _, err := os.Stat(w.ModelPath()); err != nil {
		t.Fatal(err)
	}
	dealRows, err := readDealCount(w.DealTest())
	if err != nil {
		t.Fatal(err)
	}
	if dealRows == 0 {
		t.Fatal("empty deal output")
	}
	if dealRows > cfg.TestUsers*cfg.DeckSize {
		t.Fatalf("deal rows %d too many", dealRows)
	}
}

func readDealCount(path string) (int, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return 0, err
	}
	n := 0
	for _, line := range splitLines(string(b)) {
		if line == "" || line == "User\tItem\tTag\tScore\trank" {
			continue
		}
		n++
	}
	return n, nil
}

func splitLines(s string) []string {
	var out []string
	start := 0
	for i := 0; i < len(s); i++ {
		if s[i] == '\n' {
			line := s[start:i]
			if len(line) > 0 && line[len(line)-1] == '\r' {
				line = line[:len(line)-1]
			}
			out = append(out, line)
			start = i + 1
		}
	}
	if start < len(s) {
		out = append(out, s[start:])
	}
	return out
}

func TestSmokeCLIWorkspace(t *testing.T) {
	// 确保默认 out 路径可创建（不跑 CLI 子进程，仅测目录契约）
	root := filepath.Join(t.TempDir(), "nested", "smoke")
	w, err := recsys.NewWorkspace(root)
	if err != nil {
		t.Fatal(err)
	}
	for _, p := range []string{w.RecallDir(), w.RankDir(), w.ModelsDir(), w.DealDir()} {
		if st, err := os.Stat(p); err != nil || !st.IsDir() {
			t.Fatalf("missing dir %s: %v", p, err)
		}
	}
}
