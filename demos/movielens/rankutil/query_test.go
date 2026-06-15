package rankutil_test

import (
	"path/filepath"
	"testing"

	"github.com/linkerlin/leaves/data"
	"github.com/linkerlin/leaves/demos/movielens/rankutil"
)

func TestGroupSliceAndRankGroup(t *testing.T) {
	path := filepath.Join("..", "..", "..", "testdata", "rank_smoke_train.tsv")
	dm, err := data.LoadRankingTSV(path, "\t")
	if err != nil {
		t.Fatal(err)
	}
	groups := dm.Groups()
	if len(groups) < 2 {
		t.Fatalf("need >=2 groups, got %d", len(groups))
	}

	start, count, err := rankutil.GroupSlice(dm, 0)
	if err != nil {
		t.Fatal(err)
	}
	if count != groups[0] || start != 0 {
		t.Fatalf("group0 start=%d count=%d want 0,%d", start, count, groups[0])
	}

	preds := make([]float64, dm.NumRow())
	for i := range preds {
		preds[i] = float64(i)
	}
	items, err := rankutil.RankGroup(dm, preds, 0, 3)
	if err != nil {
		t.Fatal(err)
	}
	if len(items) != 3 {
		t.Fatalf("topk=3 got %d items", len(items))
	}
	if items[0].Score < items[1].Score {
		t.Fatal("expected descending scores")
	}
}

func TestGroupQID(t *testing.T) {
	if got := rankutil.GroupQID(2, 60); got != 62 {
		t.Fatalf("GroupQID=%d want 62", got)
	}
}
