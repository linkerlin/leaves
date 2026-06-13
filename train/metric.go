package train

import (
	"fmt"

	"github.com/dmitryikh/leaves/data"
	"github.com/dmitryikh/leaves/metrics"
)

func evalMetricFor(cfg Config, numGroups int) (metrics.Metric, error) {
	name := cfg.EvalMetric
	if name == "" {
		switch cfg.Objective {
		case ObjectiveBinaryLogistic:
			name = EvalAUC
		case ObjectiveMultiSoftmax, ObjectiveMultiSoftprob:
			name = EvalMLogLoss
		case ObjectiveRankPairwise, ObjectiveRankNDCG:
			if cfg.NDCGK > 0 {
				name = fmt.Sprintf("ndcg@%d", cfg.NDCGK)
			} else {
				name = EvalNDCG
			}
		case ObjectivePoisson, ObjectiveGamma:
			name = EvalRMSE
		default:
			name = EvalRMSE
		}
	}
	groups := metricGroups(cfg)
	opt := metrics.Options{NumClass: numGroups, Groups: groups, NDCGK: cfg.NDCGK}
	return metrics.Resolve(name, opt)
}

func metricGroups(cfg Config) []int {
	if cfg.EvalSet == nil {
		return nil
	}
	if gm, ok := cfg.EvalSet.(data.GroupedMatrix); ok {
		return gm.Groups()
	}
	return nil
}

func evaluateTrainMetric(l *Learner, labels, preds []float64, dm data.Matrix) (float64, error) {
	if l.metric == nil {
		return 0, nil
	}
	groups := groupsFromMatrix(dm)
	return metrics.Evaluate(l.metric, labels, preds, groups)
}

func groupsFromMatrix(dm data.Matrix) []int {
	if gm, ok := dm.(data.GroupedMatrix); ok {
		return gm.Groups()
	}
	return nil
}
