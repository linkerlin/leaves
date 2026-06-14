package objective

import "fmt"

type factory func(numClass int) (Func, error)

var extra = map[string]factory{}

// Register 注册自定义目标（T5 插件化预备；内置目标仍走 switch）。
func Register(name string, f factory) {
	extra[name] = f
}

// ByNameWithClass 解析目标函数（多分类需 numClass）。
// 排序目标使用默认 RankOptions；训练侧通过 ConfigureRanking 覆盖。
func ByNameWithClass(name string, numClass int) (Func, error) {
	if f, ok := extra[name]; ok {
		return f(numClass)
	}
	switch name {
	case "reg:squarederror", "":
		return SquaredError{}, nil
	case "binary:logistic":
		return BinaryLogistic{}, nil
	case "multi:softmax":
		if numClass < 2 {
			return nil, fmt.Errorf("objective: multi:softmax needs num_class >= 2")
		}
		return Multiclass{NumClass: numClass}, nil
	case "multi:softprob":
		if numClass < 2 {
			return nil, fmt.Errorf("objective: multi:softprob needs num_class >= 2")
		}
		return Multiclass{NumClass: numClass, Softprob: true}, nil
	case "reg:gamma":
		return Gamma{}, nil
	case "count:poisson":
		return Poisson{}, nil
	case "rank:pairwise":
		return NewRankPairwise(RankTrainConfig{
			PairMethod:          RankPairTopK,
			NumPairPerSample:    defaultTopKPairs,
			LambdaNormalization: true,
		}), nil
	case "rank:ndcg":
		return NewRankNDCG(RankTrainConfig{
			LambdaNorm:          true,
			PairMethod:          RankPairTopK,
			NumPairPerSample:    defaultTopKPairs,
			LambdaNormalization: true,
		}), nil
	case "rank:listwise":
		return NewRankListwise(RankTrainConfig{}), nil
	default:
		return nil, fmt.Errorf("objective: unsupported %q", name)
	}
}

// ConfigureRanking 用训练超参覆盖排序目标选项。
func ConfigureRanking(obj Func, cfg RankTrainConfig) Func {
	switch obj.(type) {
	case RankPairwise:
		return NewRankPairwise(cfg)
	case RankNDCG:
		return NewRankNDCG(cfg)
	case RankListwise:
		return NewRankListwise(cfg)
	default:
		return obj
	}
}

// IsMulticlass 判断是否为多分类目标。
func IsMulticlass(obj Func) (*Multiclass, bool) {
	m, ok := obj.(Multiclass)
	if !ok {
		return nil, false
	}
	return &m, true
}
