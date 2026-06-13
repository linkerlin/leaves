package objective

// RankOptionsFromTrain 由训练配置构造 RankOptions（避免 objective→train 循环依赖）。
type RankTrainConfig struct {
	NDCGK       int
	LambdaNorm  bool
	MaxPosition int
}

func NewRankPairwise(cfg RankTrainConfig) RankPairwise {
	return RankPairwise{Opts: RankOptions{Scale: RankScalePairwise, MaxPosition: cfg.MaxPosition}}
}

func NewRankNDCG(cfg RankTrainConfig) RankNDCG {
	return RankNDCG{Opts: RankOptions{
		Scale:       RankScaleNDCG,
		NDCGK:       cfg.NDCGK,
		Norm:        cfg.LambdaNorm,
		MaxPosition: cfg.MaxPosition,
	}}
}

func NewRankListwise(cfg RankTrainConfig) RankListwise {
	_ = cfg
	return RankListwise{}
}
