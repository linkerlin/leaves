package objective

import (
	"fmt"
	"math"

	"github.com/linkerlin/leaves/data"
)

// AFTNormal survival:aft（指数 AFT，μ=exp(pred)）。
// 标量标签：正=事件时间，负=-右删失时间；区间删失见 data.AFTIntervalMatrix。
type AFTNormal struct{}

func (AFTNormal) Name() string { return "survival:aft" }

func (AFTNormal) GradHess(pred, label, weight float64) (float64, float64) {
	iv := data.AFTIntervalFromScalarLabel(label)
	return aftIntervalGradHess(pred, weight, iv)
}

func (AFTNormal) InitialPred(labels []float64, weights []float64) float64 {
	mean := weightedMeanPositive(labels, weights)
	if mean <= 0 {
		return 0
	}
	return math.Log(mean)
}

func (AFTNormal) GradHessBatch(preds, labels, weights, grad, hess []float64) error {
	n := len(preds)
	if len(labels) != n || len(grad) != n || len(hess) != n {
		return fmt.Errorf("objective: aft length mismatch")
	}
	intervals := make([]data.AFTInterval, n)
	for i, y := range labels {
		intervals[i] = data.AFTIntervalFromScalarLabel(y)
	}
	return aftGradHessInterval(preds, weights, intervals, grad, hess)
}

// GradHessInterval 区间删失 batch 梯度。
func (AFTNormal) GradHessInterval(preds, weights []float64, intervals []data.AFTInterval, grad, hess []float64) error {
	return aftGradHessInterval(preds, weights, intervals, grad, hess)
}

func aftGradHessInterval(preds, weights []float64, intervals []data.AFTInterval, grad, hess []float64) error {
	n := len(preds)
	if len(intervals) != n || len(grad) != n || len(hess) != n {
		return fmt.Errorf("objective: aft interval length mismatch")
	}
	for i := 0; i < n; i++ {
		w := 1.0
		if weights != nil && i < len(weights) {
			w = weights[i]
		}
		g, h := aftIntervalGradHess(preds[i], w, intervals[i])
		grad[i] = g
		hess[i] = h
	}
	return nil
}

func aftIntervalGradHess(pred, w float64, iv data.AFTInterval) (float64, float64) {
	mu := math.Exp(pred)
	L, U := iv.Lower, iv.Upper

	if math.IsInf(U, 1) {
		g := -w * L / mu
		h := w * math.Max(L/mu, 1e-16)
		return g, h
	}
	if L == U {
		y := L
		g := w * (1 - y/mu)
		h := w * math.Max(y/mu, 1e-16)
		return g, h
	}
	if L == 0 {
		aU := math.Exp(-U / mu)
		F := 1 - aU
		if F < 1e-16 {
			F = 1e-16
		}
		dF := aU * U / mu
		g := -w * dF / F
		h := w * math.Max(math.Abs(g), 1e-16)
		return g, h
	}
	aL := math.Exp(-L / mu)
	aU := math.Exp(-U / mu)
	diff := aL - aU
	if diff < 1e-16 {
		diff = 1e-16
	}
	dDiff := aL*L/mu - aU*U/mu
	g := -w * dDiff / diff
	h := w * math.Max(math.Abs(g), 1e-16)
	return g, h
}

// IsAFT 判断是否为 AFT 目标。
func IsAFT(obj Func) (AFTNormal, bool) {
	a, ok := obj.(AFTNormal)
	return a, ok
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
