// Package linear 提供 gblinear 模型的中间表示与推理引擎。
package linear

// LinearIR XGBoost gblinear 模型的中间表示。
// Weights 布局与 XGBoost 一致：output k 的预测为
//
//	base_score + Weights[numFeatures*numOutputGroups + k]
//	           + Σ_i fvals[i] * Weights[numOutputGroups*i + k]
type LinearIR struct {
	NumFeatures     int
	NumOutputGroups int
	BaseScore       float64
	Weights         []float64
	Name            string
}

// NEstimators gblinear 恒为 1。
func (l *LinearIR) NEstimators() int { return 1 }
