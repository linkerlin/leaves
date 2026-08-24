package deal

import (
	"testing"

	"github.com/linkerlin/leaves/v2/recsys"
)

func TestRunDedupAndTagLimit(t *testing.T) {
	scored := []recsys.ManifestRow{
		{User: "u1", Item: "a", Tag: "x", Score: 3},
		{User: "u1", Item: "b", Tag: "x", Score: 2.5},
		{User: "u1", Item: "c", Tag: "x", Score: 2.4},
		{User: "u1", Item: "d", Tag: "y", Score: 2.0},
		{User: "u1", Item: "e", Tag: "y", Score: 1.0},
		{User: "u2", Item: "a", Tag: "x", Score: 9},
	}
	recent := map[string]map[string]struct{}{
		"u1": {"a": {}},
	}
	rows, logs, err := Run(scored, recent, Config{DeckSize: 3, MaxSameTag: 1})
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) == 0 {
		t.Fatal("empty deck")
	}
	var u1 []recsys.DealRow
	for _, r := range rows {
		if r.User == "u1" {
			u1 = append(u1, r)
		}
	}
	if len(u1) != 3 {
		t.Fatalf("u1 deck=%d want 3: %+v", len(u1), u1)
	}
	for _, r := range u1 {
		if r.Item == "a" {
			t.Fatal("recent item a should be dropped")
		}
	}
	if len(logs) != 2 {
		t.Fatalf("logs=%d want 2", len(logs))
	}
}

func TestRecentItems(t *testing.T) {
	got := RecentItems([]recsys.Interaction{
		{User: "u", Item: "i1"},
		{User: "u", Item: "i2"},
	})
	if _, ok := got["u"]["i1"]; !ok {
		t.Fatal("missing i1")
	}
}
