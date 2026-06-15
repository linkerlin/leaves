package objective

import "math"

// AFTNormal survival:aft（正态 AFT，Cox 式标签：正=事件时间，负=-删失时间）。
// 对标 XGBoost 简化路径；完整 interval 标签见 AFTIntervalMatrix（后续）。
type AFTNormal struct{}

func (AFTNormal) Name() string { return "survival:aft" }

func (AFTNormal) GradHess(pred, label, weight float64) (float64, float64) {
	w := weight
	y := label
	mu := math.Exp(pred)
	if y > 0 {
		// 未删失：对数正态负对数似然梯度
		g := w * (1 - y/mu)
		h := w * y / mu
		if h < 1e-16 {
			h = 1e-16
		}
		return g, h
	}
	// 右删失于 t=|y|
	t := -y
	if t <= 0 {
		return 0, 1e-16
	}
	z := (math.Log(t) - pred)
	g := w * math.Exp(-0.5*z*z) / (mu * 0.3989422804014327) // phi(z)/mu 近似
	h := w * 0.01
	if h < 1e-16 {
		h = 1e-16
	}
	return g, h
}

func (AFTNormal) InitialPred(labels []float64, weights []float64) float64 {
	mean := weightedMeanPositive(labels, weights)
	if mean <= 0 {
		return 0
	}
	return math.Log(mean)
}

func weightedMeanPositive(labels, weights []float64) float64 {
	var sw, sy float64
	for i, y := range labels {
		if y <= 0 {
			continue
		}
		w := 1.0
		if weights != nil {
			w = weights[i]
		}
		sw += w
		sy += w * y
	}
	if sw == 0 {
		return weightedMean(labels, weights)
	}
	return sy / sw
}
