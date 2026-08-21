package prep

import (
	"fmt"
	"sort"
	"time"

	"github.com/linkerlin/leaves/v2/recsys"
	"github.com/linkerlin/leaves/v2/recsys/contract"
	"github.com/linkerlin/leaves/v2/recsys/split"
	"github.com/linkerlin/leaves/v2/recsys/synth"
	"github.com/linkerlin/leaves/v2/recsys/tsvio"
)

// Result 数据准备产物。
type Result struct {
	Report   recsys.PrepReport
	UserQIDs []recsys.UserQID
}

// Run 清洗、切分、写盘。
func Run(w recsys.Workspace, ds synth.Dataset) (Result, error) {
	train, test := synth.SplitInteractions(ds.Raw, ds.TrainUsers, ds.TestUsers)

	dedupTrain, dropTrain := dedupeLatest(train)
	dedupTest, dropTest := dedupeLatest(test)

	userQIDs := assignQIDs(ds.TrainUsers, ds.TestUsers)

	report := recsys.PrepReport{
		Stage:       "data-prep",
		SplitMode:   "user",
		TrainUsers:  len(ds.TrainUsers),
		TestUsers:   len(ds.TestUsers),
		TrainRows:   len(dedupTrain),
		TestRows:    len(dedupTest),
		CatalogSize: len(ds.Catalog),
		TagVocab:    synth.TagVocab(ds.Catalog),
		Dropped: map[string]int{
			"duplicate_user_item_train": dropTrain,
			"duplicate_user_item_test":  dropTest,
		},
	}

	if err := tsvio.WriteInteractions(w.SamplesTrain(), dedupTrain); err != nil {
		return Result{}, err
	}
	if err := tsvio.WriteInteractions(w.SamplesTest(), dedupTest); err != nil {
		return Result{}, err
	}
	if err := tsvio.WriteCatalog(w.ItemsCatalog(), ds.FeatNames, ds.Catalog); err != nil {
		return Result{}, err
	}
	if err := tsvio.WriteUserQIDs(w.UserQID(), userQIDs); err != nil {
		return Result{}, err
	}
	if err := tsvio.WriteJSON(w.PrepReport(), report); err != nil {
		return Result{}, err
	}
	return Result{Report: report, UserQIDs: userQIDs}, nil
}

// RunTimeSplit 时间切分主线（RC 遗留收口）：要求全部交互带 UTC 时间戳。
// train=[…,TrainEnd)、[TrainEnd,ValStart) 隔离带丢弃、val 不进四段流水线、
// test=[TestStart,…)；切后经 CheckLeakage as-of 门禁。cfg 全零时按 70/72/85
// 分位自动推导（split.SuggestTimeConfig）。同一用户可同时出现在 train 与
// test（时间切分正常语义；QID 按 user+split 唯一）。
func RunTimeSplit(w recsys.Workspace, ds synth.Dataset, cfg split.TimeConfig) (Result, error) {
	events := make([]contract.InteractionEvent, len(ds.Raw))
	for i, r := range ds.Raw {
		if r.Time.IsZero() {
			return Result{}, fmt.Errorf("prep: time-split requires UTC timestamps on all interactions (row %d user=%s item=%s missing); fill Time or use user split", i, r.User, r.Item)
		}
		if r.Time.Location() != time.UTC {
			return Result{}, fmt.Errorf("prep: interaction time must be UTC (row %d got %s)", i, r.Time.Location())
		}
		events[i] = contract.InteractionEvent{
			EventID:    fmt.Sprintf("e%06d", i),
			OccurredAt: r.Time,
			SubjectID:  r.User,
			ItemID:     r.Item,
			EventType:  contract.EventRating,
			Value:      r.Score,
			Source:     "prep",
		}
	}
	if cfg.TrainEnd.IsZero() && cfg.ValStart.IsZero() && cfg.TestStart.IsZero() {
		suggested, err := split.SuggestTimeConfig(events)
		if err != nil {
			return Result{}, err
		}
		cfg = suggested
	}
	trainEv, valEv, testEv, err := split.Split(events, cfg)
	if err != nil {
		return Result{}, err
	}
	if err := split.CheckLeakage(trainEv, cfg.TestStart); err != nil {
		return Result{}, err
	}

	train, test := split.Assign(trainEv), split.Assign(testEv)
	// split.Assign 只映射四元中的 User/Item/Score；Tag 是物品目录属性，从 catalog 回填
	// （否则空 Tag 会让 samples TSV 行尾列丢失）。
	tagOf := make(map[string]string, len(ds.Catalog))
	for _, it := range ds.Catalog {
		tagOf[it.Item] = it.Tag
	}
	for _, rows := range [][]recsys.Interaction{train, test} {
		for i := range rows {
			rows[i].Tag = tagOf[rows[i].Item]
		}
	}
	dedupTrain, dropTrain := dedupeLatest(train)
	dedupTest, dropTest := dedupeLatest(test)
	trainUsers, testUsers := usersOf(dedupTrain), usersOf(dedupTest)
	userQIDs := assignQIDs(trainUsers, testUsers)
	isolated := len(events) - len(trainEv) - len(valEv) - len(testEv)

	report := recsys.PrepReport{
		Stage:       "data-prep",
		SplitMode:   "time",
		TrainUsers:  len(trainUsers),
		TestUsers:   len(testUsers),
		TrainRows:   len(dedupTrain),
		TestRows:    len(dedupTest),
		CatalogSize: len(ds.Catalog),
		TagVocab:    synth.TagVocab(ds.Catalog),
		Dropped: map[string]int{
			"duplicate_user_item_train": dropTrain,
			"duplicate_user_item_test":  dropTest,
			"time_isolated":             isolated,
			"time_val_unused":           len(valEv),
		},
	}

	if err := tsvio.WriteInteractions(w.SamplesTrain(), dedupTrain); err != nil {
		return Result{}, err
	}
	if err := tsvio.WriteInteractions(w.SamplesTest(), dedupTest); err != nil {
		return Result{}, err
	}
	if err := tsvio.WriteCatalog(w.ItemsCatalog(), ds.FeatNames, ds.Catalog); err != nil {
		return Result{}, err
	}
	if err := tsvio.WriteUserQIDs(w.UserQID(), userQIDs); err != nil {
		return Result{}, err
	}
	if err := tsvio.WriteJSON(w.PrepReport(), report); err != nil {
		return Result{}, err
	}
	return Result{Report: report, UserQIDs: userQIDs}, nil
}

// usersOf 按字典序返回去重用户列表（确定性 QID 分配的前提）。
func usersOf(rows []recsys.Interaction) []string {
	seen := make(map[string]struct{}, len(rows))
	out := make([]string, 0, len(rows))
	for _, r := range rows {
		if _, ok := seen[r.User]; !ok {
			seen[r.User] = struct{}{}
			out = append(out, r.User)
		}
	}
	sort.Strings(out)
	return out
}

func dedupeLatest(rows []recsys.Interaction) ([]recsys.Interaction, int) {
	best := map[string]recsys.Interaction{}
	dropped := 0
	for _, r := range rows {
		key := r.User + "\x00" + r.Item
		if prev, ok := best[key]; ok {
			dropped++
			if r.Score > prev.Score {
				best[key] = r
			}
			continue
		}
		best[key] = r
	}
	out := make([]recsys.Interaction, 0, len(best))
	for _, v := range best {
		out = append(out, v)
	}
	return out, dropped
}

func assignQIDs(trainUsers, testUsers []string) []recsys.UserQID {
	out := make([]recsys.UserQID, 0, len(trainUsers)+len(testUsers))
	for i, u := range trainUsers {
		out = append(out, recsys.UserQID{User: u, QID: i, Split: "train"})
	}
	base := len(trainUsers)
	for i, u := range testUsers {
		out = append(out, recsys.UserQID{User: u, QID: base + i, Split: "test"})
	}
	return out
}

// ValidateCatalogCoverage 校验 catalog 覆盖 samples 中全部 Item。
func ValidateCatalogCoverage(samples []recsys.Interaction, catalog []recsys.CatalogItem) error {
	have := map[string]struct{}{}
	for _, it := range catalog {
		have[it.Item] = struct{}{}
	}
	for _, r := range samples {
		if _, ok := have[r.Item]; !ok {
			return fmt.Errorf("prep: catalog missing item %s", r.Item)
		}
	}
	return nil
}
