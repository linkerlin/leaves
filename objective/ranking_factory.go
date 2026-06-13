package objective

// RankOptionsFromTrain 由训练配置构造 RankOptions（避免 objective→train 循环依赖）。
type RankTrainConfig struct {
	NDCGK                  int
	LambdaNorm             bool
	MaxPosition            int
	PairMethod             RankPairMethod
	NumPairPerSample       int
	PairSeed               int64
	LambdaNormalization    bool
	ScoreNormalization     bool
}

func rankOptsFromTrain(cfg RankTrainConfig) RankOptions {
	return RankOptions{
		MaxPosition:         cfg.MaxPosition,
		PairMethod:          cfg.PairMethod,
		NumPairPerSample:    cfg.NumPairPerSample,
		PairSeed:            cfg.PairSeed,
		LambdaNormalization: cfg.LambdaNormalization,
		ScoreNormalization:  cfg.ScoreNormalization,
	}
}

func NewRankPairwise(cfg RankTrainConfig) RankPairwise {
	o := rankOptsFromTrain(cfg)
	o.Scale = RankScalePairwise
	return RankPairwise{Opts: o}
}

func NewRankNDCG(cfg RankTrainConfig) RankNDCG {
	o := rankOptsFromTrain(cfg)
	o.Scale = RankScaleNDCG
	o.NDCGK = cfg.NDCGK
	o.Norm = cfg.LambdaNorm
	return RankNDCG{Opts: o}
}

func NewRankListwise(cfg RankTrainConfig) RankListwise {
	_ = cfg
	return RankListwise{}
}
