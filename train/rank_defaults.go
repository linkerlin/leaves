package train

const defaultLambdaRankNumPairPerSample = 32

func applyRankPairDefaults(cfg *Config) {
	if cfg == nil || !isRankObjective(cfg.Objective) {
		return
	}
	if cfg.LambdaRankPairMethod == "" {
		cfg.LambdaRankPairMethod = LambdaRankPairTopK
	}
	if cfg.LambdaRankNumPairPerSample <= 0 {
		cfg.LambdaRankNumPairPerSample = defaultLambdaRankNumPairPerSample
	}
	// 显式 full 走经典全配对（无 λ 归一化）；其余策略默认开启。
	if cfg.LambdaRankPairMethod != LambdaRankPairFull {
		cfg.LambdaRankNormalization = true
	}
}
