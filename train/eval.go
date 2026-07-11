package train

import (
	"fmt"

	"github.com/linkerlin/leaves/data"
)

// Eval 在 Matrix 上计算配置的 EvalMetric（需已 Fit）。
func (l *Learner) Eval(dm data.Matrix) (float64, error) {
	if l.booster == nil {
		return 0, fmt.Errorf("train: not fitted")
	}
	if dm == nil {
		return 0, fmt.Errorf("train: nil matrix")
	}
	metric := l.metric
	if metric == nil {
		var err error
		metric, err = evalMetricFor(l.cfg, l.numGroups)
		if err != nil {
			return 0, err
		}
	}
	n := dm.NumRow()
	g := l.numGroups
	preds := make([]float64, n*g)
	l.predictMarginsInternal(dm, preds, false)
	labels, metricPreds := metricInputs(l.cfg, labelsForMetric(l.cfg, dm), preds, g)
	return evaluateTrainMetric(&Learner{cfg: l.cfg, metric: metric, numGroups: l.numGroups}, labels, metricPreds, dm)
}

// labelsForMetric 多目标用扁平 Targets，否则 Labels。
func labelsForMetric(cfg Config, dm data.Matrix) []float64 {
	if cfg.NumTarget > 1 {
		if mt, ok := data.AsMultiTarget(dm); ok {
			return mt.Targets()
		}
	}
	return dm.Labels()
}
