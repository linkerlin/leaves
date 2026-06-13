package booster

import "github.com/dmitryikh/leaves/data"

// Booster 训练期 booster 接口。
type Booster interface {
	NumOutputGroups() int
	Boost(dm data.Matrix, grad, hess []float64)
	PredictMargins(dm data.Matrix, out []float64)
}
