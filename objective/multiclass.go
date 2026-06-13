package objective

import (
	"math"
)

// Multiclass 多分类目标（multi:softmax / multi:softprob）。
type Multiclass struct {
	NumClass int
	Softprob bool
}

func (m Multiclass) Name() string {
	if m.Softprob {
		return "multi:softprob"
	}
	return "multi:softmax"
}

func (m Multiclass) Classes() int {
	if m.NumClass <= 0 {
		return 2
	}
	return m.NumClass
}

func (m Multiclass) GradHess(pred, label, weight float64) (float64, float64) {
	return 0, 0
}

func (m Multiclass) InitialPred(labels []float64, weights []float64) float64 {
	return 0
}

// GradHessVec 写入 grad/hess 切片（长度 = NumClass）。
func (m Multiclass) GradHessVec(pred []float64, label float64, weight float64, grad, hess []float64) {
	k := m.Classes()
	probs := softmax(pred, k)
	cls := int(label)
	if cls < 0 || cls >= k {
		cls = 0
	}
	for c := 0; c < k; c++ {
		y := 0.0
		if c == cls {
			y = 1.0
		}
		p := probs[c]
		grad[c] = (p - y) * weight
		h := 2 * p * (1 - p) * weight
		if h < 1e-16 {
			h = 1e-16
		}
		hess[c] = h
	}
}

// InitialPredVec 返回每类初始 margin（均匀先验 log(1/k)）。
func (m Multiclass) InitialPredVec(labels []float64, weights []float64) []float64 {
	k := m.Classes()
	out := make([]float64, k)
	v := -math.Log(float64(k))
	for i := range out {
		out[i] = v
	}
	return out
}

func softmax(pred []float64, k int) []float64 {
	out := make([]float64, k)
	maxV := pred[0]
	for c := 1; c < k; c++ {
		if c < len(pred) && pred[c] > maxV {
			maxV = pred[c]
		}
	}
	sum := 0.0
	for c := 0; c < k; c++ {
		v := 0.0
		if c < len(pred) {
			v = pred[c]
		}
		e := math.Exp(v - maxV)
		out[c] = e
		sum += e
	}
	if sum > 0 {
		inv := 1 / sum
		for c := range out {
			out[c] *= inv
		}
	}
	return out
}

// Gamma reg:gamma（log link，pred 为 margin）。
type Gamma struct{}

func (Gamma) Name() string { return "reg:gamma" }

func (Gamma) GradHess(pred, label, weight float64) (float64, float64) {
	if label <= 0 {
		label = 1e-16
	}
	w := weight
	mu := math.Exp(pred)
	g := (1 - label/math.Max(mu, 1e-16)) * w
	h := label / math.Max(mu, 1e-16) * w
	if h < 1e-16 {
		h = 1e-16
	}
	return g, h
}

func (Gamma) InitialPred(labels []float64, weights []float64) float64 {
	mean := weightedMean(labels, weights)
	if mean <= 0 {
		return 0
	}
	return math.Log(mean)
}

// Poisson count:poisson（log link）。
type Poisson struct{}

func (Poisson) Name() string { return "count:poisson" }

func (Poisson) GradHess(pred, label, weight float64) (float64, float64) {
	w := weight
	mu := math.Exp(pred)
	g := (mu - label) * w
	h := math.Max(mu, 1e-16) * w
	return g, h
}

func (Poisson) InitialPred(labels []float64, weights []float64) float64 {
	mean := weightedMean(labels, weights)
	if mean <= 0 {
		return 0
	}
	return math.Log(mean)
}

func weightedMean(labels []float64, weights []float64) float64 {
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
