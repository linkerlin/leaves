package rankconv

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/linkerlin/leaves/v2/recsys"
)

func TestRunWritesTSV(t *testing.T) {
	dir := t.TempDir()
	rankPath := filepath.Join(dir, "rank.tsv")
	manPath := filepath.Join(dir, "man.jsonl")
	recall := []recsys.RecallRow{
		{User: "u1", Item: "i1", Tag: "a", RecallScore: 0.9, Feats: []float64{1, 2}},
		{User: "u1", Item: "i2", Tag: "b", RecallScore: 0.1, Feats: []float64{3, 4}},
	}
	samples := []recsys.Interaction{{User: "u1", Item: "i1", Score: 1}}
	qids := []recsys.UserQID{{User: "u1", QID: 7, Split: "train"}}
	res, err := Run(recall, samples, qids, "train", rankPath, manPath)
	if err != nil {
		t.Fatal(err)
	}
	if res.Rows != 2 {
		t.Fatalf("rows=%d want 2", res.Rows)
	}
	if len(res.Manifest) != 2 {
		t.Fatalf("manifest=%d", len(res.Manifest))
	}
	b, err := os.ReadFile(rankPath)
	if err != nil {
		t.Fatal(err)
	}
	if len(b) == 0 {
		t.Fatal("empty tsv")
	}
}
