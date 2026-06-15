package booster

import (
	"math/rand"

	"github.com/linkerlin/leaves/tree"
)

// DARTConfig DART 训练参数。
type DARTConfig struct {
	RateDrop      float64 // 丢弃已有树概率
	SkipDrop      int     // 前 N 棵树不参与丢弃
	NormalizeType string  // "tree" 或 "forest"
}

// ApplyDARTDrop 本轮新增树后对历史树做 dropout。
func ApplyDARTDrop(f *tree.ForestIR, dart DARTConfig, numNewTrees int, rng *rand.Rand) {
	if f == nil || dart.RateDrop <= 0 || len(f.Trees) <= numNewTrees {
		return
	}
	if len(f.WeightDrop) < len(f.Trees) {
		f.WeightDrop = append(f.WeightDrop, make([]float64, len(f.Trees)-len(f.WeightDrop))...)
		for i := range f.WeightDrop {
			if f.WeightDrop[i] == 0 {
				f.WeightDrop[i] = 1.0
			}
		}
	}
	end := len(f.Trees) - numNewTrees
	if end <= dart.SkipDrop {
		return
	}
	beforeSum := 0.0
	for i := 0; i < end; i++ {
		beforeSum += f.WeightDrop[i]
	}
	dropped := 0
	for i := dart.SkipDrop; i < end; i++ {
		if rng.Float64() < dart.RateDrop {
			f.WeightDrop[i] = 0
			dropped++
		}
	}
	if dropped == 0 {
		return
	}
	if dart.NormalizeType == "forest" && beforeSum > 0 {
		afterSum := 0.0
		for i := 0; i < end; i++ {
			afterSum += f.WeightDrop[i]
		}
		if afterSum > 0 {
			scale := beforeSum / afterSum
			for i := 0; i < len(f.WeightDrop); i++ {
				if f.WeightDrop[i] > 0 {
					f.WeightDrop[i] *= scale
				}
			}
		}
	}
	f.Name = "leaves.dart"
}
