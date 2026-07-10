package metrics

type metricFactory func(Options) (Metric, error)

var extraMetrics = map[string]metricFactory{}

// Register 注册自定义评估指标（名称经 NormalizeName）。
// 内置指标见 builtins.go；ndcg@K / map@K 由 Resolve 拆 K 后查 "ndcg"/"map"。
func Register(name string, f metricFactory) {
	extraMetrics[NormalizeName(name)] = f
}
