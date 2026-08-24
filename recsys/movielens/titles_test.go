package movielens

import (
	"path/filepath"
	"testing"
)

func TestTitlesRoundTrip(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "meta", "item_titles.tsv")
	in := map[string]string{"7": "Seven", "1": "Toy\tStory"}
	if err := WriteTitles(p, in); err != nil {
		t.Fatal(err)
	}
	out, err := LoadTitles(p)
	if err != nil {
		t.Fatal(err)
	}
	if out["7"] != "Seven" {
		t.Fatalf("%v", out)
	}
	if out["1"] != "Toy Story" {
		t.Fatalf("tab should collapse: %q", out["1"])
	}
}
