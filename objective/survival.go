package objective

import (
	"fmt"
	"math"
	"sort"
)

// SurvivalFunc 生存分析目标：全批梯度（Cox 等）。
type SurvivalFunc interface {
	Func
	GradHessBatch(preds, labels, weights, grad, hess []float64) error
}

// Cox survival:cox（Breslow 偏似然，对标 XGBoost）。
// 标签：正数=事件时间，负数=-删失时间。
type Cox struct{}

func (Cox) Name() string { return "survival:cox" }

func (Cox) GradHess(pred, label, weight float64) (float64, float64) {
	_ = pred
	_ = label
	_ = weight
	return 0, 0
}

func (Cox) InitialPred(_ []float64, _ []float64) float64 { return 0 }

func (Cox) GradHessBatch(preds, labels, weights, grad, hess []float64) error {
	n := len(preds)
	if len(labels) != n || len(grad) != n || len(hess) != n {
		return fmt.Errorf("objective: cox length mismatch")
	}
	order := make([]int, n)
	for i := range order {
		order[i] = i
	}
	sort.Slice(order, func(i, j int) bool {
		ai := math.Abs(labels[order[i]])
		aj := math.Abs(labels[order[j]])
		if ai != aj {
			return ai < aj
		}
		return order[i] < order[j]
	})

	expPSum := 0.0
	for _, idx := range order {
		expPSum += math.Exp(preds[idx])
	}

	var rK, sK float64
	var lastExpP, lastAbsY float64
	var accumulatedSum float64

	for _, idx := range order {
		p := preds[idx]
		expP := math.Exp(p)
		w := 1.0
		if weights != nil && idx < len(weights) {
			w = weights[idx]
		}
		y := labels[idx]
		absY := math.Abs(y)

		accumulatedSum += lastExpP
		if lastAbsY < absY {
			expPSum -= accumulatedSum
			accumulatedSum = 0
		} else if lastAbsY > absY {
			return fmt.Errorf("objective: cox labels must be sortable by |time|")
		}

		if y > 0 {
			rK += 1.0 / expPSum
			sK += 1.0 / (expPSum * expPSum)
		}

		g := expP*rK - boolToFloat(y > 0)
		h := expP*rK - expP*expP*sK
		if h < 1e-16 {
			h = 1e-16
		}
		grad[idx] = g * w
		hess[idx] = h * w

		lastAbsY = absY
		lastExpP = expP
	}
	return nil
}

func boolToFloat(b bool) float64 {
	if b {
		return 1
	}
	return 0
}

// IsSurvival 判断是否为生存分析目标。
func IsSurvival(obj Func) (SurvivalFunc, bool) {
	s, ok := obj.(SurvivalFunc)
	if !ok {
		return nil, false
	}
	return s, true
}

// GradHessSurvival 计算生存目标全批梯度。
func GradHessSurvival(s SurvivalFunc, preds, labels, weights, grad, hess []float64) error {
	return s.GradHessBatch(preds, labels, weights, grad, hess)
}
