package objective

import (
	"math"
)

const defaultTweediePower = 1.5

// Tweedie reg:tweedie（log link，对标 XGBoost）。
type Tweedie struct {
	VariancePower float64 // (1, 2)，默认 1.5
}

func NewTweedie(power float64) Tweedie {
	if power <= 1 || power >= 2 {
		power = defaultTweediePower
	}
	return Tweedie{VariancePower: power}
}

func (t Tweedie) Name() string { return "reg:tweedie" }

func (t Tweedie) power() float64 {
	if t.VariancePower > 1 && t.VariancePower < 2 {
		return t.VariancePower
	}
	return defaultTweediePower
}

func (t Tweedie) GradHess(pred, label, weight float64) (float64, float64) {
	if label < 0 {
		label = 0
	}
	rho := t.power()
	w := weight
	grad := -label*math.Exp((1-rho)*pred) + math.Exp((2-rho)*pred)
	hess := -label*(1-rho)*math.Exp((1-rho)*pred) + (2-rho)*math.Exp((2-rho)*pred)
	if hess < 1e-16 {
		hess = 1e-16
	}
	return grad * w, hess * w
}

func (t Tweedie) InitialPred(labels []float64, weights []float64) float64 {
	mean := weightedMean(labels, weights)
	if mean <= 0 {
		return 0
	}
	return math.Log(mean)
}

// ConfigureTweedie 用训练超参覆盖方差幂。
func ConfigureTweedie(obj Func, power float64) Func {
	if _, ok := obj.(Tweedie); ok {
		return NewTweedie(power)
	}
	return obj
}
