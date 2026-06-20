package prep

import (
	"fmt"

	"github.com/linkerlin/leaves/recsys"
	"github.com/linkerlin/leaves/recsys/synth"
	"github.com/linkerlin/leaves/recsys/tsvio"
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
