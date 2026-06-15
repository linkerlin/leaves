package train

import (
	"fmt"

	"github.com/linkerlin/leaves/booster"
)

// LearningRateScheduler 按 boosting 轮次调整学习率（对标 XGBoost learning_rate schedule）。
type LearningRateScheduler interface {
	// Rate 返回第 round 轮（0-based）使用的学习率；base 为 Config 初始 LearningRate。
	Rate(round int, base float64) float64
}

// TrainingCallback 训练循环钩子（每轮 Boost 与 metric 评估之后调用）。
type TrainingCallback interface {
	AfterIteration(ctx *CallbackContext) error
}

// CallbackContext 回调上下文。
type CallbackContext struct {
	Round         int     // 0-based
	LearningRate  float64 // 本轮实际学习率
	TrainMetric   float64
	TrainMetricOK bool
	EvalMetric    float64 // EvalSet 非空且 EvalMetric 可算时为验证分数
	EvalMetricOK  bool
	Learner       *Learner
}

// FuncCallback 函数式回调。
type FuncCallback func(ctx *CallbackContext) error

func (f FuncCallback) AfterIteration(ctx *CallbackContext) error { return f(ctx) }

// ExponentialLRScheduler 指数衰减：base * gamma^round。
func ExponentialLRScheduler(gamma float64) LearningRateScheduler {
	if gamma <= 0 {
		gamma = 1
	}
	return expLR{gamma: gamma}
}

type expLR struct{ gamma float64 }

func (s expLR) Rate(round int, base float64) float64 {
	if round <= 0 {
		return base
	}
	r := base
	for i := 0; i < round; i++ {
		r *= s.gamma
	}
	return r
}

// StepLRScheduler 阶梯衰减：每 step 轮乘以 factor。
func StepLRScheduler(step int, factor float64) LearningRateScheduler {
	if step <= 0 {
		step = 1
	}
	if factor <= 0 {
		factor = 1
	}
	return stepLR{step: step, factor: factor}
}

type stepLR struct {
	step   int
	factor float64
}

func (s stepLR) Rate(round int, base float64) float64 {
	if round <= 0 {
		return base
	}
	steps := round / s.step
	r := base
	for i := 0; i < steps; i++ {
		r *= s.factor
	}
	return r
}

func (l *Learner) onRoundStart(round int) {
	if l.cfg.LRScheduler == nil {
		return
	}
	lr := l.cfg.LRScheduler.Rate(round, l.baseLearningRate)
	l.cfg.LearningRate = lr
	l.syncBoosterLearningRate(lr)
}

func (l *Learner) onRoundEnd(round int, trainMetric float64, trainMetricOK bool, evalMetric float64, evalMetricOK bool) error {
	if len(l.cfg.Callbacks) == 0 {
		return nil
	}
	ctx := &CallbackContext{
		Round:         round,
		LearningRate:  l.cfg.LearningRate,
		TrainMetric:   trainMetric,
		TrainMetricOK: trainMetricOK,
		EvalMetric:    evalMetric,
		EvalMetricOK:  evalMetricOK,
		Learner:       l,
	}
	for _, cb := range l.cfg.Callbacks {
		if cb == nil {
			continue
		}
		if err := cb.AfterIteration(ctx); err != nil {
			return fmt.Errorf("train: callback round %d: %w", round, err)
		}
	}
	return nil
}

func (l *Learner) syncBoosterLearningRate(lr float64) {
	booster.SetLearningRate(l.booster, lr)
}
