package objective

import "fmt"

func init() {
	// 标量 / 生存
	Register("reg:squarederror", func(int) (Func, error) { return SquaredError{}, nil })
	Register("", func(int) (Func, error) { return SquaredError{}, nil })
	Register("binary:logistic", func(int) (Func, error) { return BinaryLogistic{}, nil })
	Register("reg:gamma", func(int) (Func, error) { return Gamma{}, nil })
	Register("count:poisson", func(int) (Func, error) { return Poisson{}, nil })
	Register("reg:tweedie", func(int) (Func, error) { return NewTweedie(defaultTweediePower), nil })
	Register("survival:cox", func(int) (Func, error) { return Cox{}, nil })
	Register("survival:aft", func(int) (Func, error) { return AFTNormal{}, nil })

	// 多分类（需 numClass）
	Register("multi:softmax", multiFactory(false))
	Register("multi:softprob", multiFactory(true))

	// 排序（默认 RankTrainConfig；ConfigureRanking 可覆盖）
	Register("rank:pairwise", func(int) (Func, error) {
		return NewRankPairwise(RankTrainConfig{
			PairMethod:          RankPairTopK,
			NumPairPerSample:    defaultTopKPairs,
			LambdaNormalization: true,
		}), nil
	})
	Register("rank:ndcg", func(int) (Func, error) {
		return NewRankNDCG(RankTrainConfig{
			LambdaNorm:          true,
			PairMethod:          RankPairTopK,
			NumPairPerSample:    defaultTopKPairs,
			LambdaNormalization: true,
		}), nil
	})
	Register("rank:listwise", func(int) (Func, error) {
		return NewRankListwise(RankTrainConfig{}), nil
	})
}

func multiFactory(softprob bool) factory {
	return func(numClass int) (Func, error) {
		if numClass < 2 {
			return nil, fmt.Errorf("objective: multi:* needs num_class >= 2")
		}
		return Multiclass{NumClass: numClass, Softprob: softprob}, nil
	}
}
