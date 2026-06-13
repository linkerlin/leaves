package objective

import (
	"math"
)

// Func 目标函数：梯度 / Hessian（逐样本）。
type Func interface {
	Name() string
	GradHess(pred, label, weight float64) (grad, hess float64)
	InitialPred(labels []float64, weights []float64) float64
}

// SquaredError reg:squarederror。
type SquaredError struct{}

func (SquaredError) Name() string { return "reg:squarederror" }

func (SquaredError) GradHess(pred, label, weight float64) (float64, float64) {
	g := (pred - label) * weight
	return g, weight
}

func (SquaredError) InitialPred(labels []float64, weights []float64) float64 {
	var sw, sy float64
	for i, y := range labels {
		w := 1.0
		if weights != nil {
			w = weights[i]
		}
		sw += w
		sy += w * y
	}
	if sw == 0 {
		return 0
	}
	return sy / sw
}

// BinaryLogistic binary:logistic（pred 为 margin）。
type BinaryLogistic struct{}

func (BinaryLogistic) Name() string { return "binary:logistic" }

func (BinaryLogistic) GradHess(pred, label, weight float64) (float64, float64) {
	p := sigmoid(pred)
	g := (p - label) * weight
	h := p * (1 - p) * weight
	if h < 1e-16 {
		h = 1e-16
	}
	return g, h
}

func (BinaryLogistic) InitialPred(labels []float64, weights []float64) float64 {
	var posW, totalW float64
	for i, y := range labels {
		w := 1.0
		if weights != nil {
			w = weights[i]
		}
		totalW += w
		if y > 0.5 {
			posW += w
		}
	}
	if totalW == 0 || posW <= 0 || posW >= totalW {
		return 0
	}
	return math.Log((posW / totalW) / (1 - posW/totalW))
}

func sigmoid(x float64) float64 {
	if x >= 0 {
		z := math.Exp(-x)
		return 1 / (1 + z)
	}
	z := math.Exp(x)
	return z / (1 + z)
}

// ByName 解析目标函数名。
func ByName(name string) (Func, error) {
	return ByNameWithClass(name, 0)
}
