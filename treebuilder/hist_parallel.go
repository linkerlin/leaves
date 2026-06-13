package treebuilder

import (
	"runtime"
	"sync"

	"github.com/dmitryikh/leaves/data"
)

func effectiveThreads(n int) int {
	if n <= 0 {
		n = runtime.NumCPU()
	}
	if n < 1 {
		n = 1
	}
	return n
}

type histSplitPick struct {
	feat  int
	thr   float64
	gain  float64
	left  []int
	right []int
	ok    bool
}

func betterHistPick(a, b histSplitPick) histSplitPick {
	if !b.ok {
		return a
	}
	if !a.ok || b.gain > a.gain {
		return b
	}
	if b.gain < a.gain {
		return a
	}
	if b.feat < a.feat {
		return b
	}
	if b.feat > a.feat {
		return a
	}
	if b.thr < a.thr {
		return b
	}
	return a
}

func findBestHistSplit(
	dm data.Matrix,
	idx []int,
	feats []int,
	grad, hess []float64,
	sumG, sumH float64,
	row []float64,
	cfg Config,
) histSplitPick {
	best := histSplitPick{gain: cfg.Gamma}
	nThreads := effectiveThreads(cfg.NumThreads)
	if nThreads <= 1 || len(feats) < 4 {
		for _, f := range feats {
			feat, thr, gain, left, right := bestHistSplit(dm, idx, f, grad, hess, sumG, sumH, row, cfg)
			if gain <= cfg.Gamma {
				continue
			}
			cand := histSplitPick{feat: feat, thr: thr, gain: gain, left: left, right: right, ok: true}
			best = betterHistPick(best, cand)
		}
		return best
	}

	var mu sync.Mutex
	var wg sync.WaitGroup
	chunk := (len(feats) + nThreads - 1) / nThreads
	for t := 0; t < nThreads; t++ {
		start := t * chunk
		if start >= len(feats) {
			break
		}
		end := start + chunk
		if end > len(feats) {
			end = len(feats)
		}
		wg.Add(1)
		go func(featBlock []int) {
			defer wg.Done()
			local := histSplitPick{gain: cfg.Gamma}
			for _, f := range featBlock {
				feat, thr, gain, left, right := bestHistSplit(dm, idx, f, grad, hess, sumG, sumH, row, cfg)
				if gain <= cfg.Gamma {
					continue
				}
				cand := histSplitPick{feat: feat, thr: thr, gain: gain, left: left, right: right, ok: true}
				local = betterHistPick(local, cand)
			}
			if !local.ok {
				return
			}
			mu.Lock()
			best = betterHistPick(best, local)
			mu.Unlock()
		}(feats[start:end])
	}
	wg.Wait()
	return best
}
