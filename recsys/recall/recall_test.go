package recall

import (
	"testing"

	"github.com/linkerlin/leaves/v2/recsys"
)

func TestRunPerUserCount(t *testing.T) {
	catalog := []recsys.CatalogItem{
		{Item: "i1", Tag: "a", Feats: []float64{1}},
		{Item: "i2", Tag: "a", Feats: []float64{2}},
		{Item: "i3", Tag: "b", Feats: []float64{3}},
		{Item: "i4", Tag: "b", Feats: []float64{4}},
		{Item: "i5", Tag: "c", Feats: []float64{5}},
	}
	samples := []recsys.Interaction{
		{User: "u1", Item: "i1", Tag: "a", Score: 1},
		{User: "u1", Item: "i3", Tag: "b", Score: 1},
	}
	qids := []recsys.UserQID{{User: "u1", QID: 1, Split: "train"}}
	rows, err := Run("train", samples, catalog, []string{"f0"}, qids, Config{PerUser: 4, MaxKnown: 1})
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != 4 {
		t.Fatalf("got %d rows want 4", len(rows))
	}
	seen := map[string]bool{}
	for _, r := range rows {
		if r.User != "u1" {
			t.Fatalf("user=%s", r.User)
		}
		if seen[r.Item] {
			t.Fatalf("dup item %s", r.Item)
		}
		seen[r.Item] = true
	}
}

func TestRunNoUsers(t *testing.T) {
	_, err := Run("test", nil, nil, nil, nil, Config{PerUser: 10})
	if err == nil {
		t.Fatal("expected error")
	}
}
