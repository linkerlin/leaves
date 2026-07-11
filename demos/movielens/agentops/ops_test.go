package agentops_test

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/linkerlin/leaves/demos/movielens/agentops"
)

func TestStatusWhenDataPresent(t *testing.T) {
	// 从仓库根或 demos 子目录运行
	_ = os.Chdir(findRoot(t))
	res := agentops.Status()
	if !res.OK {
		t.Fatalf("status: %+v", res)
	}
	if res.Data["data_dir"] == nil {
		t.Fatal("missing data_dir")
	}
	// 数据应已在仓库中
	if res.Data["train_ready"] != true {
		t.Skip("rank_movielens_train.tsv not present; run gen_rank_movielens.py")
	}
}

func TestTrainEvalRecommendSmoke(t *testing.T) {
	if testing.Short() {
		t.Skip("short")
	}
	_ = os.Chdir(findRoot(t))
	st := agentops.Status()
	if st.Data["train_ready"] != true {
		t.Skip("need MovieLens ranking TSV")
	}
	// 小轮数 smoke
	tr := agentops.Train(agentops.TrainParams{
		Objective: "rank:ndcg",
		Rounds:    8,
		Depth:     3,
		LR:        0.15,
	})
	if !tr.OK {
		t.Fatalf("train: %+v", tr)
	}
	model, _ := tr.Data["model"].(string)
	ev := agentops.Eval(model, 10)
	if !ev.OK {
		t.Fatalf("eval: %+v", ev)
	}
	rc := agentops.Recommend(agentops.RecommendParams{Model: model, Group: 0, TopK: 5, QID: -1})
	if !rc.OK {
		t.Fatalf("recommend: %+v", rc)
	}
	items, _ := rc.Data["items"].([]map[string]any)
	if items == nil {
		// JSON unmarshaling of interface - check differently
		raw, ok := rc.Data["items"]
		if !ok {
			t.Fatal("no items")
		}
		arr, ok := raw.([]any)
		if !ok || len(arr) == 0 {
			t.Fatalf("items empty: %T %v", raw, raw)
		}
	}
}

func findRoot(t *testing.T) string {
	t.Helper()
	cwd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	dir := cwd
	for i := 0; i < 8; i++ {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			if _, err := os.Stat(filepath.Join(dir, "testdata", "rank_movielens_train.tsv")); err == nil {
				return dir
			}
			// go.mod found but no data — still return for Status test
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
