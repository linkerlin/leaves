package data

import (
	"math/rand"
)

// SubsampleIndices 按 rate 无放回抽样行索引；rate>=1 返回全量。
func SubsampleIndices(n int, rate float64, rng *rand.Rand) []int {
	if n <= 0 {
		return nil
	}
	if rate >= 1.0 || rate <= 0 {
		idx := make([]int, n)
		for i := range idx {
			idx[i] = i
		}
		return idx
	}
	k := int(float64(n) * rate)
	if k < 1 {
		k = 1
	}
	if k >= n {
		idx := make([]int, n)
		for i := range idx {
			idx[i] = i
		}
		return idx
	}
	perm := rng.Perm(n)
	out := make([]int, k)
	copy(out, perm[:k])
	return out
}

// ColsampleIndices 按 rate 无放回抽样特征列；rate>=1 返回全列。
func ColsampleIndices(ncols int, rate float64, rng *rand.Rand) []int {
	if ncols <= 0 {
		return nil
	}
	if rate >= 1.0 || rate <= 0 {
		out := make([]int, ncols)
		for i := range out {
			out[i] = i
		}
		return out
	}
	k := int(float64(ncols) * rate)
	if k < 1 {
		k = 1
	}
	if k >= ncols {
		out := make([]int, ncols)
		for i := range out {
			out[i] = i
		}
		return out
	}
	perm := rng.Perm(ncols)
	out := make([]int, k)
	for i := 0; i < k; i++ {
		out[i] = perm[i]
	}
	return out
}
