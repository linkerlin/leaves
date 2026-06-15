package train

import (
	"fmt"

	"github.com/linkerlin/leaves/data"
	"github.com/linkerlin/leaves/metrics"
)

// EarlyStopping 早停配置。
type EarlyStopping struct {
	Rounds    int
	Maximize  bool
	BestScore float64
	bestRound int
	noImprove int
}

// NewEarlyStopping 创建早停器。
func NewEarlyStopping(rounds int, maximize bool) *EarlyStopping {
	if rounds <= 0 {
		rounds = 10
	}
	es := &EarlyStopping{Rounds: rounds, Maximize: maximize}
	if maximize {
		es.BestScore = -1e300
	} else {
		es.BestScore = 1e300
	}
	return es
}

func (es *EarlyStopping) update(score float64, round int) bool {
	if es == nil {
		return false
	}
	improved := false
	if es.Maximize {
		if score > es.BestScore {
			es.BestScore = score
			es.bestRound = round
			es.noImprove = 0
			improved = true
		}
	} else if score < es.BestScore {
		es.BestScore = score
		es.bestRound = round
		es.noImprove = 0
		improved = true
	}
	if !improved {
		es.noImprove++
	}
	return es.noImprove >= es.Rounds
}

// BestRound 返回最优轮次（1-based）。
func (es *EarlyStopping) BestRound() int { return es.bestRound }

func evalMetricOnSet(l *Learner, dm data.Matrix) (float64, error) {
	if l.metric == nil || dm == nil {
		return 0, fmt.Errorf("no metric or eval set")
	}
	n := dm.NumRow()
	g := l.numGroups
	preds := make([]float64, n*g)
	if err := l.PredictMargins(dm, preds); err != nil {
		return 0, err
	}
	labels, metricPreds := metricInputs(l.cfg, dm.Labels(), preds, g)
	return metrics.Evaluate(l.metric, labels, metricPreds, groupsFromMatrix(dm))
}

func metricMaximize(m metrics.Metric) bool {
	if m == nil {
		return false
	}
	return m.HigherIsBetter()
}
