package model

import (
	"github.com/linkerlin/leaves/metrics"
)

// Evaluate 用指定指标评估预测结果。
func Evaluate(m metrics.Metric, yTrue, yPred []float64) (float64, error) {
	return m.Evaluate(yTrue, yPred)
}

// EvaluatePerGroup 按 query/group 切片评估（排序指标或点态指标组内平均）。
func EvaluatePerGroup(m metrics.Metric, yTrue, yPred []float64, groups []int) (float64, error) {
	return m.EvaluatePerGroup(yTrue, yPred, groups)
}
