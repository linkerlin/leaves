package pipeline

import (
	"fmt"
	"math/rand"
	"sort"
	"time"

	"github.com/linkerlin/leaves/v2/recsys"
	"github.com/linkerlin/leaves/v2/recsys/deal"
	"github.com/linkerlin/leaves/v2/recsys/prep"
	"github.com/linkerlin/leaves/v2/recsys/rankconv"
	"github.com/linkerlin/leaves/v2/recsys/recall"
	"github.com/linkerlin/leaves/v2/recsys/split"
	"github.com/linkerlin/leaves/v2/recsys/synth"
	"github.com/linkerlin/leaves/v2/recsys/trainrank"
	"github.com/linkerlin/leaves/v2/recsys/tsvio"
)

// Result 流水线各阶段摘要。
type Result struct {
	Prep        recsys.PrepReport    `json:"prep"`
	RecallTrain int                  `json:"recall_train"`
	RecallTest  int                  `json:"recall_test"`
	RankTrain   int                  `json:"rank_train"`
	RankTest    int                  `json:"rank_test"`
	Eval        trainrank.EvalResult `json:"eval"`
	DealRows    int                  `json:"deal_rows"`
}

// LoadDatasetFromFiles 从自定义四元交互 TSV 与物品目录 TSV 构建 Dataset（WP-28）。
func LoadDatasetFromFiles(interactionsPath, catalogPath string, seed int64, testUserRatio float64) (synth.Dataset, error) {
	raw, err := tsvio.ReadInteractions(interactionsPath)
	if err != nil {
		return synth.Dataset{}, fmt.Errorf("pipeline: read interactions: %w", err)
	}
	if len(raw) == 0 {
		return synth.Dataset{}, fmt.Errorf("pipeline: empty interactions in %s", interactionsPath)
	}
	featNames, catalog, err := tsvio.ReadCatalog(catalogPath)
	if err != nil {
		return synth.Dataset{}, fmt.Errorf("pipeline: read catalog: %w", err)
	}
	if len(catalog) == 0 {
		return synth.Dataset{}, fmt.Errorf("pipeline: empty catalog in %s", catalogPath)
	}
	userSet := make(map[string]bool)
	var users []string
	for _, r := range raw {
		if !userSet[r.User] {
			userSet[r.User] = true
			users = append(users, r.User)
		}
	}
	sort.Strings(users)
	if len(users) < 2 {
		return synth.Dataset{}, fmt.Errorf("pipeline: need at least 2 distinct users, got %d", len(users))
	}
	rng := rand.New(rand.NewSource(seed))
	rng.Shuffle(len(users), func(i, j int) { users[i], users[j] = users[j], users[i] })
	if testUserRatio <= 0 || testUserRatio >= 1 {
		testUserRatio = 0.25
	}
	nTest := int(float64(len(users))*testUserRatio + 0.5)
	if nTest < 1 {
		nTest = 1
	}
	if nTest >= len(users) {
		nTest = len(users) - 1
	}
	nTrain := len(users) - nTest
	trainUsers := append([]string(nil), users[:nTrain]...)
	testUsers := append([]string(nil), users[nTrain:]...)
	sort.Strings(trainUsers)
	sort.Strings(testUsers)

	base := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
	for i := range raw {
		if raw[i].Time.IsZero() {
			raw[i].Time = base.Add(time.Duration(i) * time.Minute)
		}
	}
	return synth.Dataset{
		Raw:        raw,
		Catalog:    catalog,
		FeatNames:  featNames,
		TrainUsers: trainUsers,
		TestUsers:  testUsers,
	}, nil
}

// Run 合成数据端到端推荐 smoke 流水线。
func Run(w recsys.Workspace, cfg recsys.SmokeConfig) (Result, error) {
	ds, err := synth.Generate(cfg)
	if err != nil {
		return Result{}, fmt.Errorf("pipeline: synth: %w", err)
	}
	return RunFromDataset(w, ds, cfg)
}

// RunFromDataset 对已有 Dataset（合成或 MovieLens 等）执行：
// prep → recall → rankconv → trainrank → deal。
func RunFromDataset(w recsys.Workspace, ds synth.Dataset, cfg recsys.SmokeConfig) (Result, error) {
	if cfg.RecallSize <= 0 {
		cfg.RecallSize = 100
	}
	if cfg.DeckSize <= 0 {
		cfg.DeckSize = 10
	}
	if cfg.MaxSameTag <= 0 {
		cfg.MaxSameTag = 3
	}
	if cfg.TrainRounds <= 0 {
		cfg.TrainRounds = 25
	}
	if cfg.NDCGK <= 0 {
		cfg.NDCGK = 10
	}
	if len(ds.Catalog) < cfg.RecallSize {
		return Result{}, fmt.Errorf("pipeline: catalog size %d < recall size %d", len(ds.Catalog), cfg.RecallSize)
	}

	// 切分模式：""/"user"=用户切分（默认）；"time"=时间切分（边界零值→自动推导）。
	var prepRes prep.Result
	var err error
	switch cfg.SplitMode {
	case "", "user":
		prepRes, err = prep.Run(w, ds)
	case "time":
		prepRes, err = prep.RunTimeSplit(w, ds, split.TimeConfig{})
	default:
		return Result{}, fmt.Errorf("pipeline: unknown split mode %q (want user|time)", cfg.SplitMode)
	}
	if err != nil {
		return Result{}, fmt.Errorf("pipeline: prep: %w", err)
	}
	if err := prep.ValidateCatalogCoverage(ds.Raw, ds.Catalog); err != nil {
		return Result{}, err
	}

	trainSamples, err := tsvio.ReadInteractions(w.SamplesTrain())
	if err != nil {
		return Result{}, err
	}
	testSamples, err := tsvio.ReadInteractions(w.SamplesTest())
	if err != nil {
		return Result{}, err
	}
	_, catalog, err := tsvio.ReadCatalog(w.ItemsCatalog())
	if err != nil {
		return Result{}, err
	}

	// 召回：部分已交互（LTR 标签）+ 未交互补齐（发牌 recent 过滤后仍有候选）
	recallCfg := recall.Config{PerUser: cfg.RecallSize}
	recallTrain, err := recall.Run("train", trainSamples, catalog, ds.FeatNames, prepRes.UserQIDs, recallCfg)
	if err != nil {
		return Result{}, fmt.Errorf("pipeline: recall train: %w", err)
	}
	if err := recall.Validate(recallTrain, cfg.RecallSize); err != nil {
		return Result{}, err
	}
	recallTest, err := recall.Run("test", testSamples, catalog, ds.FeatNames, prepRes.UserQIDs, recallCfg)
	if err != nil {
		return Result{}, fmt.Errorf("pipeline: recall test: %w", err)
	}
	if err := recall.Validate(recallTest, cfg.RecallSize); err != nil {
		return Result{}, err
	}
	if err := tsvio.WriteRecall(w.RecallTrain(), ds.FeatNames, recallTrain); err != nil {
		return Result{}, err
	}
	if err := tsvio.WriteRecall(w.RecallTest(), ds.FeatNames, recallTest); err != nil {
		return Result{}, err
	}

	rankTrainRes, err := rankconv.Run(recallTrain, trainSamples, prepRes.UserQIDs, "train", w.RankTrain(), w.RankTrainManifest())
	if err != nil {
		return Result{}, fmt.Errorf("pipeline: rankconv train: %w", err)
	}
	rankTestRes, err := rankconv.Run(recallTest, testSamples, prepRes.UserQIDs, "test", w.RankTest(), w.RankTestManifest())
	if err != nil {
		return Result{}, fmt.Errorf("pipeline: rankconv test: %w", err)
	}

	eval, err := trainrank.Run(w, cfg)
	if err != nil {
		return Result{}, fmt.Errorf("pipeline: trainrank: %w", err)
	}

	scored, err := tsvio.ReadManifestJSONL(w.RankTestScored())
	if err != nil {
		return Result{}, err
	}
	recent := deal.RecentItems(testSamples)
	dealCfg := deal.Config{DeckSize: cfg.DeckSize, MaxSameTag: cfg.MaxSameTag}
	dealRows, logs, err := deal.Run(scored, recent, dealCfg)
	if err != nil {
		return Result{}, fmt.Errorf("pipeline: deal: %w", err)
	}
	if err := deal.Validate(dealRows, recent, cfg.MaxSameTag, cfg.DeckSize); err != nil {
		return Result{}, err
	}
	if err := tsvio.WriteDeal(w.DealTest(), dealRows); err != nil {
		return Result{}, err
	}
	if err := deal.WriteLog(w.DealLog(), logs); err != nil {
		return Result{}, err
	}

	return Result{
		Prep:        prepRes.Report,
		RecallTrain: len(recallTrain),
		RecallTest:  len(recallTest),
		RankTrain:   rankTrainRes.Rows,
		RankTest:    rankTestRes.Rows,
		Eval:        eval,
		DealRows:    len(dealRows),
	}, nil
}
