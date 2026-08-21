package prep_test

import (
	"os"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/linkerlin/leaves/v2/recsys"
	"github.com/linkerlin/leaves/v2/recsys/prep"
	"github.com/linkerlin/leaves/v2/recsys/split"
	"github.com/linkerlin/leaves/v2/recsys/synth"
)

func TestPrepTimeSplitAuto(t *testing.T) {
	ds, err := synth.Generate(recsys.DefaultSmokeConfig())
	if err != nil {
		t.Fatal(err)
	}
	w, err := recsys.NewWorkspace(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}

	res, err := prep.RunTimeSplit(w, ds, split.TimeConfig{})
	if err != nil {
		t.Fatal(err)
	}
	r := res.Report
	if r.SplitMode != "time" {
		t.Fatalf("split mode: got %q want time", r.SplitMode)
	}
	if r.TrainRows == 0 || r.TestRows == 0 {
		t.Fatalf("empty split: train=%d test=%d", r.TrainRows, r.TestRows)
	}

	// 行数对账：train + test + 重复丢弃 + 隔离带 + val == 原始行数
	dup := r.Dropped["duplicate_user_item_train"] + r.Dropped["duplicate_user_item_test"]
	used := r.TrainRows + r.TestRows + dup + r.Dropped["time_isolated"] + r.Dropped["time_val_unused"]
	if used != len(ds.Raw) {
		t.Fatalf("row accounting: used=%d want %d", used, len(ds.Raw))
	}

	for _, p := range []string{w.SamplesTrain(), w.SamplesTest(), w.UserQID(), w.PrepReport()} {
		if _, err := os.Stat(p); err != nil {
			t.Fatal(err)
		}
	}

	// 时间切分语义：同一用户允许同时出现在 train 与 test（正常重叠）
	inTrain, overlap := map[string]bool{}, 0
	for _, q := range res.UserQIDs {
		if q.Split == "train" {
			inTrain[q.User] = true
		}
	}
	for _, q := range res.UserQIDs {
		if q.Split == "test" && inTrain[q.User] {
			overlap++
		}
	}
	if overlap == 0 {
		t.Fatal("expected train/test user overlap in time split")
	}

	// 确定性：同数据重跑报告一致（用户排序保证 QID 分配稳定）
	res2, err := prep.RunTimeSplit(w, ds, split.TimeConfig{})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(res.Report, res2.Report) {
		t.Fatalf("non-deterministic report:\n%+v\nvs\n%+v", res.Report, res2.Report)
	}
}

func TestPrepTimeSplitRequiresTimestamps(t *testing.T) {
	ds, err := synth.Generate(recsys.DefaultSmokeConfig())
	if err != nil {
		t.Fatal(err)
	}
	for i := range ds.Raw {
		ds.Raw[i].Time = time.Time{}
	}
	w, err := recsys.NewWorkspace(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	if _, err := prep.RunTimeSplit(w, ds, split.TimeConfig{}); err == nil || !strings.Contains(err.Error(), "timestamps") {
		t.Fatalf("want timestamp error, got %v", err)
	}
}
