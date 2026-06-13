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
	depth int,
	cfg Config,
) histSplitPick {
	best := histSplitPick{gain: cfg.Gamma}
	if len(feats) == 0 {
		return best
	}

	if !gpuHistBatchEnabled(cfg) {
		return parallelEvalHistFeats(dm, idx, feats, grad, hess, sumG, sumH, row, cfg, nil)
	}

	gpuFeats := filterGPUHistFeats(feats, idx, depth, cfg)
	gpuSet := make(map[int]struct{}, len(gpuFeats))
	for _, f := range gpuFeats {
		gpuSet[f] = struct{}{}
	}
	cpuOnly := make([]int, 0, len(feats)-len(gpuFeats))
	for _, f := range feats {
		if _, onGPU := gpuSet[f]; !onGPU {
			cpuOnly = append(cpuOnly, f)
		}
	}

	var gpuDone <-chan map[int]gpuHistResult
	if len(gpuFeats) > 0 {
		gpuDone = enqueueGPUHistBatch(gpuFeats, idx, grad, hess, sumG, sumH, cfg.Lambda, cfg)
	}

	// 阶段 1：GPU worker 排队时，CPU 并行评估非 GPU 特征
	if len(cpuOnly) > 0 {
		best = betterHistPick(best, parallelEvalHistFeats(dm, idx, cpuOnly, grad, hess, sumG, sumH, row, cfg, nil))
	}

	// 阶段 2：取回 GPU 直方图，评估 GPU 特征
	if gpuDone != nil {
		prebuilt := <-gpuDone
		if len(prebuilt) > 0 {
			best = betterHistPick(best, parallelEvalHistFeats(dm, idx, gpuFeats, grad, hess, sumG, sumH, row, cfg, prebuilt))
		} else {
			best = betterHistPick(best, parallelEvalHistFeats(dm, idx, gpuFeats, grad, hess, sumG, sumH, row, cfg, nil))
		}
	}

	return best
}

func parallelEvalHistFeats(
	dm data.Matrix,
	idx []int,
	feats []int,
	grad, hess []float64,
	sumG, sumH float64,
	row []float64,
	cfg Config,
	prebuilt map[int]gpuHistResult,
) histSplitPick {
	best := histSplitPick{gain: cfg.Gamma}
	if len(feats) == 0 {
		return best
	}

	evalFeat := func(f int) histSplitPick {
		var gh *gpuHistResult
		if prebuilt != nil {
			if r, ok := prebuilt[f]; ok && r.ok {
				rr := r
				gh = &rr
			}
		}
		return histSplitFromFeat(dm, idx, f, grad, hess, sumG, sumH, row, cfg, gh)
	}

	nThreads := effectiveThreads(cfg.NumThreads)
	if nThreads <= 1 || len(feats) < 4 {
		for _, f := range feats {
			if pick := evalFeat(f); pick.ok {
				best = betterHistPick(best, pick)
			}
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
				if pick := evalFeat(f); pick.ok {
					local = betterHistPick(local, pick)
				}
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
