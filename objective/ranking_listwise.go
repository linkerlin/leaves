package objective

import "math"

// RankListwise rank:listwise（ListNet 组内 softmax 交叉熵，纯 listwise 损失）。
//
// 目标分布 q_i ∝ exp(label_i)，模型分布 p_i = softmax(pred)。
// 损失 L = -Σ q_i log(p_i)；梯度 dL/dpred_i = p_i - q_i（与 GBTree 叶子 -g/h 约定一致）。
type RankListwise struct{}

func (RankListwise) Name() string { return "rank:listwise" }

func (RankListwise) GradHess(pred, label, weight float64) (float64, float64) {
	_ = pred
	_ = label
	_ = weight
	return 0, 0
}

func (RankListwise) InitialPred(labels []float64, weights []float64) float64 {
	_ = labels
	_ = weights
	return 0
}

func (RankListwise) GradHessGroup(preds, labels, weights []float64, grad, hess []float64) {
	computeListwiseSoftmax(preds, labels, weights, grad, hess)
}

func computeListwiseSoftmax(preds, labels, weights []float64, grad, hess []float64) {
	n := len(preds)
	if n == 0 || len(labels) != n || len(grad) != n || len(hess) != n {
		return
	}
	for i := range grad {
		grad[i] = 0
		hess[i] = 0
	}
	if n == 1 {
		const minHess = 1e-16
		hess[0] = minHess
		return
	}

	q := softmaxFromValues(labels)
	p := softmaxFromValues(preds)

	const minHess = 1e-16
	for i := 0; i < n; i++ {
		w := weightAt(weights, i)
		grad[i] = (p[i] - q[i]) * w
		h := p[i] * (1 - p[i]) * w
		if h < minHess {
			h = minHess
		}
		hess[i] = h
	}
}

// softmaxFromValues 数值稳定 softmax；全零/常数时返回均匀分布。
func softmaxFromValues(v []float64) []float64 {
	n := len(v)
	out := make([]float64, n)
	if n == 0 {
		return out
	}
	maxV := v[0]
	for _, x := range v[1:] {
		if x > maxV {
			maxV = x
		}
	}
	var sum float64
	for i, x := range v {
		e := math.Exp(x - maxV)
		out[i] = e
		sum += e
	}
	if sum <= 0 {
		inv := 1 / float64(n)
		for i := range out {
			out[i] = inv
		}
		return out
	}
	inv := 1 / sum
	for i := range out {
		out[i] *= inv
	}
	return out
}

// ListwiseLoss 计算单 query 的 listwise 交叉熵（测试/调试）。
func ListwiseLoss(preds, labels []float64) float64 {
	if len(preds) == 0 || len(labels) != len(preds) {
		return 0
	}
	q := softmaxFromValues(labels)
	p := softmaxFromValues(preds)
	var loss float64
	for i := range q {
		if q[i] <= 0 {
			continue
		}
		pi := p[i]
		if pi < 1e-15 {
			pi = 1e-15
		}
		loss -= q[i] * math.Log(pi)
	}
	return loss
}
