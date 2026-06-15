package metrics

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
}
