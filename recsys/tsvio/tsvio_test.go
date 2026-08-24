package tsvio

import (
	"path/filepath"
	"testing"

	"github.com/linkerlin/leaves/v2/recsys"
)

func TestInteractionsRoundTrip(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "samples.tsv")
	in := []recsys.Interaction{
		{User: "u1", Item: "i1", Score: 1.5, Tag: "drama"},
		{User: "u2", Item: "i2", Score: 0, Tag: "comedy"},
	}
	if err := WriteInteractions(p, in); err != nil {
		t.Fatal(err)
	}
	out, err := ReadInteractions(p)
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 2 || out[0].User != "u1" || out[0].Item != "i1" || out[1].Tag != "comedy" {
		t.Fatalf("%+v", out)
	}
}

func TestCatalogRoundTrip(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "items.tsv")
	names := []string{"feat_pop", "feat_q"}
	items := []recsys.CatalogItem{
		{Item: "i1", Tag: "a", Feats: []float64{1, 2}},
	}
	if err := WriteCatalog(p, names, items); err != nil {
		t.Fatal(err)
	}
	gotNames, got, err := ReadCatalog(p)
	if err != nil {
		t.Fatal(err)
	}
	if len(gotNames) != 2 || gotNames[0] != "feat_pop" {
		t.Fatalf("names=%v", gotNames)
	}
	if len(got) != 1 || got[0].Item != "i1" || got[0].Feats[1] != 2 {
		t.Fatalf("%+v", got)
	}
}

func TestUserQIDRoundTrip(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "user_qid.tsv")
	in := []recsys.UserQID{{User: "u1", QID: 7, Split: "train"}}
	if err := WriteUserQIDs(p, in); err != nil {
		t.Fatal(err)
	}
	out, err := ReadUserQIDs(p)
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 1 || out[0].QID != 7 || out[0].Split != "train" {
		t.Fatalf("%+v", out)
	}
}

func TestManifestJSONLRoundTrip(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "man.jsonl")
	in := []recsys.ManifestRow{{User: "u", Item: "i", Tag: "t", RecallScore: 0.9, Score: 1.2}}
	if err := WriteManifestJSONL(p, in); err != nil {
		t.Fatal(err)
	}
	out, err := ReadManifestJSONL(p)
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 1 || out[0].Item != "i" {
		t.Fatalf("%+v", out)
	}
}

func TestReadInteractionsBadRow(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "bad.tsv")
	if err := writeLines(p, []string{"User\tItem\tScore\tTag", "only-two\tcols"}); err != nil {
		t.Fatal(err)
	}
	if _, err := ReadInteractions(p); err == nil {
		t.Fatal("expected error")
	}
}
