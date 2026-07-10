package metrics

import "fmt"

func init() {
	Register("rmse", func(o Options) (Metric, error) { return RMSE{}, nil })
	Register("mae", func(o Options) (Metric, error) { return MAE{}, nil })
	Register("mape", func(o Options) (Metric, error) { return MAPE{}, nil })
	Register("rmsle", func(o Options) (Metric, error) { return RMSLE{}, nil })
	Register("logloss", func(o Options) (Metric, error) { return LogLoss{}, nil })
	Register("error", func(o Options) (Metric, error) { return Error{}, nil })
	Register("binary_error", func(o Options) (Metric, error) { return Error{}, nil })
	Register("auc", func(o Options) (Metric, error) { return AUC{}, nil })
	Register("aucpr", func(o Options) (Metric, error) { return AUC{}, nil })
	Register("ndcg", func(o Options) (Metric, error) {
		return NDCG{RankingMetric: RankingMetric{Groups: o.Groups, K: o.NDCGK}}, nil
	})
	Register("map", func(o Options) (Metric, error) {
		return MAP{RankingMetric: RankingMetric{Groups: o.Groups, K: o.NDCGK}}, nil
	})
	// 多分类（需 NumClass）
	Register("mlogloss", func(o Options) (Metric, error) {
		if o.NumClass < 2 {
			return nil, fmt.Errorf("metrics: mlogloss needs num_class >= 2")
		}
		return MLogLoss{NumClass: o.NumClass}, nil
	})
	Register("merror", func(o Options) (Metric, error) {
		if o.NumClass < 2 {
			return nil, fmt.Errorf("metrics: merror needs num_class >= 2")
		}
		return MError{NumClass: o.NumClass}, nil
	})
}
