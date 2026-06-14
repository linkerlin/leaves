package train

import "testing"

func TestRankPairDefaults(t *testing.T) {
	cfg := Config{Objective: ObjectiveRankNDCG}
	applyRankPairDefaults(&cfg)
	if cfg.LambdaRankPairMethod != LambdaRankPairTopK {
		t.Fatalf("pair method: %q", cfg.LambdaRankPairMethod)
	}
	if cfg.LambdaRankNumPairPerSample != defaultLambdaRankNumPairPerSample {
		t.Fatalf("num pair: %d", cfg.LambdaRankNumPairPerSample)
	}
	if !cfg.LambdaRankNormalization {
		t.Fatal("expected lambda normalization default true for rank:ndcg")
	}
}

func TestRankPairDefaultsFullNoNormalization(t *testing.T) {
	cfg := Config{
		Objective:            ObjectiveRankPairwise,
		LambdaRankPairMethod: LambdaRankPairFull,
	}
	applyRankPairDefaults(&cfg)
	if cfg.LambdaRankNormalization {
		t.Fatal("full pairing should keep normalization off for classic path")
	}
}
