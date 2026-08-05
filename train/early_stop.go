package train

import (
	"fmt"

	"github.com/linkerlin/leaves/v2/booster"
	"github.com/linkerlin/leaves/v2/data"
	"github.com/linkerlin/leaves/v2/metrics"
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

// ApplyBestRound 将 GBTree 森林截断到早停 best_round（就地修改 booster 内 ForestIR）。
// 无早停、best_round<=0 或非树模型时为 no-op，返回实际保留的 boosting 轮数。
func (l *Learner) ApplyBestRound() int {
	br := l.BestRound()
	if br <= 0 || l.booster == nil {
		return 0
	}
	if b, ok := l.booster.(*booster.GBTree); ok {
		f := b.Forest()
		if f == nil {
			return 0
		}
		f.TruncateToNEstimators(br)
		return f.NEstimators()
	}
	return 0
}

// BoostRounds 返回当前模型 boosting 轮数（树模型）；线性模型返回 0。
func (l *Learner) BoostRounds() int {
	if l == nil || l.booster == nil {
		return 0
	}
	if b, ok := l.booster.(*booster.GBTree); ok {
		if f := b.Forest(); f != nil {
			return f.NEstimators()
		}
	}
	return 0
}

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
	labels, metricPreds := metricInputs(l.cfg, labelsForMetric(l.cfg, dm), preds, g)
	return metrics.Evaluate(l.metric, labels, metricPreds, groupsFromMatrix(dm))
}

func metricMaximize(m metrics.Metric) bool {
	if m == nil {
		return false
	}
	return m.HigherIsBetter()
}
