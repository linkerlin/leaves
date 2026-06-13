package objective

import (
	"math"
	"sort"
)

// RankPairMethod 配对策略（对标 XGBoost lambdarank_pair_method）。
type RankPairMethod int

const (
	// RankPairFull leaves 经典全配对：所有 label[i]>label[j] 文档对。
	RankPairFull RankPairMethod = iota
	// RankPairTopK 对标 XGBoost topk：按预测排序后，top-k 位置两两配对。
	RankPairTopK
	// RankPairMean 对标 XGBoost mean：按 label 分桶随机采样配对。
	RankPairMean
)

const (
	defaultTopKPairs  = 32
	defaultMeanPairs  = 1
	lambdaNormEps     = 1e-16
	scoreNormEps      = 0.01
)

type rankPairFn func(rankHi, rankLo int)

// forEachRankPair 按策略枚举 rank 位置对 (i,j)，i/j 为按预测分排序后的位置索引。
func forEachRankPair(
	preds, labels []float64,
	method RankPairMethod,
	numPair int,
	seed int64,
	groupIdx int,
	boostRound int,
	fn rankPairFn,
) {
	n := len(preds)
	if n <= 1 {
		return
	}
	rankIdx := sortedByPredIndices(preds)
	k := numPair
	if k <= 0 {
		if method == RankPairMean {
			k = defaultMeanPairs
		} else {
			k = defaultTopKPairs
		}
	}

	switch method {
	case RankPairFull:
		for i := 0; i < n; i++ {
			for j := 0; j < n; j++ {
				if labels[i] <= labels[j] {
					continue
				}
				hi, lo := docToRankPos(rankIdx, i, j)
				fn(hi, lo)
			}
		}
	case RankPairTopK:
		limit := n
		if k < limit {
			limit = k
		}
		for i := 0; i < limit; i++ {
			for j := i + 1; j < n; j++ {
				fn(i, j)
			}
		}
	case RankPairMean:
		rng := meanPairRNG(seed, groupIdx, boostRound)
		meanPairBuckets(rankIdx, labels, k, rng, fn)
	}
}

func docToRankPos(rankIdx []int, docHi, docLo int) (rankHi, rankLo int) {
	rankHi, rankLo = -1, -1
	for r, doc := range rankIdx {
		if doc == docHi {
			rankHi = r
		}
		if doc == docLo {
			rankLo = r
		}
	}
	if rankHi < 0 || rankLo < 0 {
		return 0, 0
	}
	if rankHi > rankLo {
		rankHi, rankLo = rankLo, rankHi
	}
	return rankHi, rankLo
}

func sortedByPredIndices(preds []float64) []int {
	n := len(preds)
	idx := make([]int, n)
	for i := range idx {
		idx[i] = i
	}
	sort.Slice(idx, func(a, b int) bool {
		if preds[idx[a]] == preds[idx[b]] {
			return idx[a] < idx[b]
		}
		return preds[idx[a]] > preds[idx[b]]
	})
	return idx
}

// minstdRand 对标 C++ std::minstd_rand（linear_congruential_engine 48271, 0, 2^31-1）。
type minstdRand struct {
	state uint32
}

func newMinstdRand(seed int) *minstdRand {
	// 对标 MSVC std::minstd_rand：seed==0 时使用 default_seed=1。
	if seed == 0 {
		seed = 1
	}
	return &minstdRand{state: uint32(seed)}
}

func (r *minstdRand) next() uint32 {
	r.state = r.state * 48271 % 2147483647
	return r.state
}

func (r *minstdRand) intn(n int) int {
	if n <= 0 {
		return 0
	}
	return int(r.next() % uint32(n))
}

func (r *minstdRand) discard(k int) {
	for i := 0; i < k; i++ {
		r.next()
	}
}

func meanPairRNG(seed int64, groupIdx, boostRound int) *minstdRand {
	r := newMinstdRand(boostRound)
	r.discard(groupIdx)
	return r
}

// meanPairBuckets 按 XGBoost mean 策略（见 lambdarank_obj.h MakePairs）。
func meanPairBuckets(rankIdx []int, labels []float64, numSample int, rng *minstdRand, fn rankPairFn) {
	n := len(rankIdx)
	if n <= 1 {
		return
	}
	// y_sorted_idx: 按 label 降序排列的 rank 位置
	ySorted := make([]int, n)
	for i := range ySorted {
		ySorted[i] = i
	}
	sort.Slice(ySorted, func(a, b int) bool {
		la := labels[rankIdx[a]]
		lb := labels[rankIdx[b]]
		if la == lb {
			return a < b
		}
		return la > lb
	})

	for i := 0; i < n; {
		j := i + 1
		for j < n && labels[rankIdx[ySorted[j]]] == labels[rankIdx[ySorted[i]]] {
			j++
		}
		nLeft := i
		nRight := n - j
		if nLeft+nRight == 0 {
			i = j
			continue
		}
		for s := 0; s < numSample; s++ {
			for pairIdx := i; pairIdx < j; pairIdx++ {
				ridx := rng.intn(nLeft + nRight)
				if ridx >= nLeft {
					ridx = ridx - i + j
				}
				fn(ySorted[pairIdx], ySorted[ridx])
			}
		}
		i = j
	}
}

func applyLambdaPair(
	preds, labels []float64,
	rankIdx []int,
	rankHi, rankLo int,
	scale float64,
	scoreNorm bool,
	grad, hess []float64,
) float64 {
	if rankHi < 0 || rankLo < 0 || rankHi >= len(rankIdx) || rankLo >= len(rankIdx) {
		return 0
	}
	// 确保 rankHi 为预测排序更高位、rankLo 为更低位
	if rankHi > rankLo {
		rankHi, rankLo = rankLo, rankHi
	}
	idxHi := rankIdx[rankHi]
	idxLo := rankIdx[rankLo]
	if labels[idxHi] == labels[idxLo] {
		return 0
	}
	// 交换使高相关 doc 为 idxHi
	rh, rl := rankHi, rankLo
	if labels[idxHi] < labels[idxLo] {
		rh, rl = rl, rh
		idxHi, idxLo = idxLo, idxHi
	}

	sHi, sLo := preds[idxHi], preds[idxLo]
	best, worst := preds[rankIdx[0]], preds[rankIdx[len(rankIdx)-1]]
	delta := math.Abs(scale)
	if scoreNorm && best != worst {
		delta /= math.Abs(sHi-sLo) + scoreNormEps
	}
	sig := sigmoidRank(sHi - sLo)
	lam := (sig - 1.0) * delta
	h := math.Max(sig*(1-sig), lambdaNormEps) * delta * 2.0

	grad[idxHi] += lam
	grad[idxLo] -= lam
	hess[idxHi] += h
	hess[idxLo] += h
	_ = rh
	_ = rl
	return -2.0 * lam
}

func normalizeLambdaGrad(grad, hess []float64, sumLambda float64) {
	if sumLambda <= 0 {
		return
	}
	norm := math.Log2(1.0+sumLambda) / sumLambda
	for i := range grad {
		grad[i] *= norm
		hess[i] *= norm
	}
}
