package pipeline

import (
	"fmt"

	"github.com/linkerlin/leaves/recsys"
	"github.com/linkerlin/leaves/recsys/deal"
	"github.com/linkerlin/leaves/recsys/prep"
	"github.com/linkerlin/leaves/recsys/rankconv"
	"github.com/linkerlin/leaves/recsys/recall"
	"github.com/linkerlin/leaves/recsys/synth"
	"github.com/linkerlin/leaves/recsys/trainrank"
	"github.com/linkerlin/leaves/recsys/tsvio"
)

// Result 流水线各阶段摘要。
type Result struct {
	Prep      recsys.PrepReport
	RecallTrain int
	RecallTest  int
	RankTrain   int
	RankTest    int
	Eval        trainrank.EvalResult
	DealRows    int
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

	prepRes, err := prep.Run(w, ds)
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
