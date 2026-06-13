package booster

import "github.com/dmitryikh/leaves/data"

// Booster 训练期 booster 接口。
type Booster interface {
	NumOutputGroups() int
	Boost(dm data.Matrix, grad, hess []float64)
	PredictMargins(dm data.Matrix, out []float64)
}

// SetLearningRate 更新 booster 学习率（LR scheduler 用）。
func SetLearningRate(b Booster, lr float64) {
	if s, ok := b.(interface{ setLearningRate(float64) }); ok {
		s.setLearningRate(lr)
	}
}
