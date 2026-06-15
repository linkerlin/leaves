package train

import (
	"fmt"

	"github.com/linkerlin/leaves/data"
	"github.com/linkerlin/leaves/objective"
)

func isRankObjective(name string) bool {
	switch name {
	case ObjectiveRankPairwise, ObjectiveRankNDCG, ObjectiveRankListwise:
		return true
	default:
		return false
	}
}

// lambdaRankNormDefault 对标 XGBoost：rank:ndcg 默认 lambdarank_norm=1，pairwise/listwise 不使用。
func lambdaRankNormDefault(cfg Config) bool {
	if cfg.LambdaRankNorm {
		return true
	}
	return cfg.Objective == ObjectiveRankNDCG
}

func (l *Learner) fitRanking(dm data.Matrix, rankObj objective.RankFunc) error {
	labels := dm.Labels()
	n := dm.NumRow()
	groups, err := data.GroupsFromRanking(dm)
	if err != nil {
		return err
	}

	if err := l.initBooster(dm, labels); err != nil {
		return err
	}
	l.metricHistory = nil

	preds := make([]float64, n)
	grad := make([]float64, n)
	hess := make([]float64, n)
	evalPreds := make([]float64, n)

	for round := 0; round < l.cfg.NumRound; round++ {
		l.onRoundStart(round)
		l.obj = objective.SetRankBoostRound(l.obj, round)
		l.predictMarginsInternal(dm, preds, false)
		if err := objective.GradHessRanking(rankObj, dm, groups, preds, grad, hess); err != nil {
			return err
		}
		l.booster.Boost(dm, grad, hess)
		var trainMetric float64
		var metricOK bool
		if l.metric != nil {
			l.predictMarginsInternal(dm, evalPreds, false)
			metricLabels, metricPreds := metricInputs(l.cfg, labels, evalPreds, 1)
			if v, err := evaluateTrainMetric(l, metricLabels, metricPreds, dm); err == nil {
				l.metricHistory = append(l.metricHistory, v)
				trainMetric = v
				metricOK = true
			}
		}
		var evalMetric float64
		var evalMetricOK bool
		if l.cfg.EvalSet != nil {
			if score, err := evalMetricOnSet(l, l.cfg.EvalSet); err == nil {
				evalMetric = score
				evalMetricOK = true
			}
		}
		if err := l.onRoundEnd(round, trainMetric, metricOK, evalMetric, evalMetricOK); err != nil {
			return err
		}
		if l.cfg.EvalSet != nil && l.cfg.EarlyStop != nil && evalMetricOK {
			if l.cfg.EarlyStop.update(evalMetric, round+1) {
				break
			}
		}
		if l.cfg.CheckpointEvery > 0 && l.cfg.CheckpointPath != "" && (round+1)%l.cfg.CheckpointEvery == 0 {
			_ = SaveCheckpointFile(l.cfg.CheckpointPath, round+1, l)
		}
	}
	return nil
}

// FitRanking 显式排序训练入口（需 GroupedMatrix）。
func FitRanking(cfg Config, dm data.GroupedMatrix) (*Learner, error) {
	if !isRankObjective(cfg.Objective) {
		cfg.Objective = ObjectiveRankNDCG
	}
	learner, err := NewLearner(cfg)
	if err != nil {
		return nil, err
	}
	if err := learner.Fit(dm); err != nil {
		return nil, err
	}
	return learner, nil
}

// ValidateRankingData 校验 groups 与 qid 连续性语义。
func ValidateRankingData(dm data.GroupedMatrix) error {
	if dm == nil {
		return fmt.Errorf("train: nil ranking matrix")
	}
	_, err := data.GroupsFromRanking(dm)
	return err
}
