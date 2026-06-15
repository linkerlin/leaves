package train

import (
	"fmt"

	"github.com/linkerlin/leaves/data"
	"github.com/linkerlin/leaves/objective"
)

func (l *Learner) fitSurvival(dm data.Matrix, survObj objective.SurvivalFunc) error {
	labels := dm.Labels()
	n := dm.NumRow()

	if err := l.initBooster(dm, labels); err != nil {
		return err
	}
	l.metricHistory = nil

	preds := make([]float64, n)
	grad := make([]float64, n)
	hess := make([]float64, n)
	evalPreds := make([]float64, n)

	for round := l.resumeFromRound; round < l.cfg.NumRound; round++ {
		l.onRoundStart(round)
		l.predictMarginsInternal(dm, preds, false)
		var err error
		if intervals, ok := data.AFTIntervalsOf(dm); ok {
			aft, isAFT := objective.IsAFT(l.obj)
			if !isAFT {
				return fmt.Errorf("train: interval censoring requires survival:aft")
			}
			err = aft.GradHessInterval(preds, dm.Weights(), intervals, grad, hess)
		} else {
			err = objective.GradHessSurvival(survObj, preds, labels, dm.Weights(), grad, hess)
		}
		if err != nil {
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

// FitSurvival 显式生存分析训练入口。
func FitSurvival(cfg Config, dm data.Matrix) (*Learner, error) {
	if cfg.Objective != ObjectiveSurvivalCox {
		cfg.Objective = ObjectiveSurvivalCox
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

// ValidateSurvivalData 校验 Cox 标签（可含删失）。
func ValidateSurvivalData(dm data.Matrix) error {
	if dm == nil {
		return fmt.Errorf("train: nil survival matrix")
	}
	if dm.NumRow() == 0 {
		return fmt.Errorf("train: empty survival matrix")
	}
	return nil
}
