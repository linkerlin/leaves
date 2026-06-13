package objective

import (
	"fmt"
	"math"
	"sort"
)

// RankFunc 排序目标：按 query group 计算 LambdaRank 梯度。
type RankFunc interface {
	Func
	// GradHessGroup 对单 query 写入 grad/hess（长度 = 组内样本数）。
	GradHessGroup(preds, labels, weights []float64, grad, hess []float64)
}

// RankScale 控制 Lambda 权重缩放（对标 XGBoost rank 目标族）。
type RankScale int

const (
	// RankScalePairwise rank:pairwise，|ΔZ|=1。
	RankScalePairwise RankScale = iota
	// RankScaleNDCG rank:ndcg，按 |ΔNDCG| 缩放。
	RankScaleNDCG
)

// RankOptions LambdaRank 超参。
type RankOptions struct {
	Scale       RankScale
	NDCGK       int  // 0 = 全量位置
	Norm        bool // lambdarank_norm：按 ideal_dcg 归一化
	MaxPosition int  // 0 = 不截断
}

// RankPairwise rank:pairwise（RankNet / LambdaMART，无 metric 缩放）。
type RankPairwise struct {
	Opts RankOptions
}

func (RankPairwise) Name() string { return "rank:pairwise" }

func (r RankPairwise) GradHess(pred, label, weight float64) (float64, float64) {
	_ = pred
	_ = label
	_ = weight
	return 0, 0
}

func (r RankPairwise) InitialPred(labels []float64, weights []float64) float64 {
	_ = labels
	_ = weights
	return 0
}

func (r RankPairwise) GradHessGroup(preds, labels, weights []float64, grad, hess []float64) {
	opts := r.Opts
	opts.Scale = RankScalePairwise
	computeLambdaRank(preds, labels, weights, grad, hess, opts)
}

// RankNDCG rank:ndcg（LambdaMART + NDCG 缩放）。
type RankNDCG struct {
	Opts RankOptions
}

func (r RankNDCG) Name() string { return "rank:ndcg" }

func (r RankNDCG) GradHess(pred, label, weight float64) (float64, float64) {
	_ = pred
	_ = label
	_ = weight
	return 0, 0
}

func (r RankNDCG) InitialPred(labels []float64, weights []float64) float64 {
	_ = labels
	_ = weights
	return 0
}

func (r RankNDCG) GradHessGroup(preds, labels, weights []float64, grad, hess []float64) {
	opts := r.Opts
	opts.Scale = RankScaleNDCG
	computeLambdaRank(preds, labels, weights, grad, hess, opts)
}

// IsRanking 判断是否为排序目标。
func IsRanking(obj Func) (RankFunc, bool) {
	rf, ok := obj.(RankFunc)
	return rf, ok
}

func computeLambdaRank(preds, labels, weights []float64, grad, hess []float64, opts RankOptions) {
	n := len(preds)
	if n == 0 || len(labels) != n || len(grad) != n || len(hess) != n {
		return
	}
	for i := range grad {
		grad[i] = 0
		hess[i] = 0
	}

	ranks := currentRanks(preds)
	ideal := idealDCG(labels, opts.NDCGK)
	if opts.Scale == RankScaleNDCG && opts.Norm && ideal <= 0 {
		return
	}

	for i := 0; i < n; i++ {
		wi := weightAt(weights, i)
		for j := 0; j < n; j++ {
			if labels[i] <= labels[j] {
				continue
			}
			w := wi * weightAt(weights, j)
			rho := sigmoidRank(preds[i] - preds[j])
			pairHess := rho * (1 - rho)

			scale := 1.0
			if opts.Scale == RankScaleNDCG {
				scale = deltaNDCG(labels, ranks, i, j, ideal, opts.NDCGK, opts.MaxPosition)
				if scale <= 0 {
					continue
				}
			}

			lambda := (1 - rho) * scale * w
			grad[i] -= lambda
			grad[j] += lambda
			h := pairHess * scale * w
			hess[i] += h
			hess[j] += h
		}
	}

	const minHess = 1e-16
	for i := range hess {
		if hess[i] < minHess {
			hess[i] = minHess
		}
	}
}

func currentRanks(scores []float64) []int {
	n := len(scores)
	type pair struct {
		score float64
		idx   int
	}
	ps := make([]pair, n)
	for i, s := range scores {
		ps[i] = pair{s, i}
	}
	sort.Slice(ps, func(a, b int) bool {
		if ps[a].score == ps[b].score {
			return ps[a].idx < ps[b].idx
		}
		return ps[a].score > ps[b].score
	})
	ranks := make([]int, n)
	for pos, p := range ps {
		ranks[p.idx] = pos
	}
	return ranks
}

func gain(rel float64) float64 {
	if rel <= 0 {
		return 0
	}
	return math.Pow(2, rel) - 1
}

func discountAt(pos int) float64 {
	return 1.0 / math.Log2(float64(pos)+2.0)
}

func dcgAtRanks(labels []float64, ranks []int, k int) float64 {
	n := len(labels)
	type item struct {
		gain float64
		pos  int
	}
	items := make([]item, n)
	for i := range labels {
		items[i] = item{gain(labels[i]), ranks[i]}
	}
	sort.Slice(items, func(a, b int) bool { return items[a].pos < items[b].pos })
	limit := n
	if k > 0 && k < limit {
		limit = k
	}
	sum := 0.0
	for i := 0; i < limit; i++ {
		sum += items[i].gain * discountAt(i)
	}
	return sum
}

func idealDCG(labels []float64, k int) float64 {
	sorted := make([]float64, len(labels))
	copy(sorted, labels)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i] > sorted[j] })
	ranks := make([]int, len(labels))
	for i := range ranks {
		ranks[i] = i
	}
	return dcgAtRanks(sorted, ranks, k)
}

func deltaNDCG(labels []float64, ranks []int, i, j int, ideal float64, k, maxPos int) float64 {
	if ideal <= 0 {
		return 0
	}
	posI, posJ := ranks[i], ranks[j]
	if maxPos > 0 {
		if posI >= maxPos && posJ >= maxPos {
			return 0
		}
	}
	gi, gj := gain(labels[i]), gain(labels[j])
	di, dj := discountAt(posI), discountAt(posJ)
	delta := math.Abs(gi*(dj-di) + gj*(di-dj)) / ideal
	return delta
}

func sigmoidRank(x float64) float64 {
	if x >= 0 {
		z := math.Exp(-x)
		return 1 / (1 + z)
	}
	z := math.Exp(x)
	return z / (1 + z)
}

func weightAt(weights []float64, i int) float64 {
	if weights == nil || i >= len(weights) {
		return 1
	}
	w := weights[i]
	if w <= 0 {
		return 1
	}
	return w
}

// GradHessRanking 对完整数据集（多 group）计算 LambdaRank 梯度。
func GradHessRanking(obj RankFunc, dm interface {
	NumRow() int
	Labels() []float64
	Weights() []float64
}, groups []int, preds, grad, hess []float64) error {
	if len(groups) == 0 {
		return fmt.Errorf("objective: ranking requires groups")
	}
	n := dm.NumRow()
	if sumGroups(groups) != n {
		return fmt.Errorf("objective: groups sum %d != rows %d", sumGroups(groups), n)
	}
	labels := dm.Labels()
	weights := dm.Weights()
	start := 0
	bufG := make([]float64, 0, 64)
	bufH := make([]float64, 0, 64)
	for _, gsz := range groups {
		if gsz <= 0 {
			return fmt.Errorf("objective: invalid group size %d", gsz)
		}
		end := start + gsz
		if cap(bufG) < gsz {
			bufG = make([]float64, gsz)
			bufH = make([]float64, gsz)
		} else {
			bufG = bufG[:gsz]
			bufH = bufH[:gsz]
		}
		for i := 0; i < gsz; i++ {
			bufG[i] = 0
			bufH[i] = 0
		}
		obj.GradHessGroup(preds[start:end], labels[start:end], sliceOrNil(weights, start, end), bufG, bufH)
		copy(grad[start:end], bufG)
		copy(hess[start:end], bufH)
		start = end
	}
	return nil
}

func sumGroups(g []int) int {
	s := 0
	for _, v := range g {
		s += v
	}
	return s
}

func sliceOrNil(w []float64, start, end int) []float64 {
	if w == nil {
		return nil
	}
	return w[start:end]
}
