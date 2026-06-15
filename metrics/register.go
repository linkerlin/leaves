package metrics

type metricFactory func(Options) (Metric, error)

var extraMetrics = map[string]metricFactory{}

// Register 注册自定义评估指标。
func Register(name string, f metricFactory) {
	extraMetrics[NormalizeName(name)] = f
}
