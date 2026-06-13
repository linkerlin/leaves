package tree

import "time"

// WalkStats 单样本森林树遍历统计。
type WalkStats struct {
	Trees int // 参与推理的树数
	Steps int // 内部节点访问次数（根到叶，含叶节点一步）
}

// ProfileWalkStats 统计单样本遍历所有启用树的步数（不执行 PredictDense）。
func ProfileWalkStats(f *ForestIR, fvals []float64, nEstimators int) WalkStats {
	if f == nil {
		return WalkStats{}
	}
	nEst := adjustNEstimators(f, nEstimators)
	if nEst <= 0 {
		return WalkStats{}
	}
	var stats WalkStats
	if len(f.IterationIndptr) > 1 {
		for iter := 0; iter < nEst; iter++ {
			if iter+1 >= len(f.IterationIndptr) {
				break
			}
			for ti := f.IterationIndptr[iter]; ti < f.IterationIndptr[iter+1]; ti++ {
				if ti >= len(f.Trees) {
					continue
				}
				stats.Steps += walkTreeSteps(&f.Trees[ti], fvals)
				stats.Trees++
			}
		}
		return stats
	}
	g := f.NumOutputGroups
	if g <= 0 {
		g = 1
	}
	for i := 0; i < nEst; i++ {
		for k := 0; k < g; k++ {
			treeIdx := i*g + k
			if treeIdx >= len(f.Trees) {
				continue
			}
			stats.Steps += walkTreeSteps(&f.Trees[treeIdx], fvals)
			stats.Trees++
		}
	}
	return stats
}

// DenseProfile 批量稠密预测 profiling 结果。
type DenseProfile struct {
	Rows            int
	Elapsed         time.Duration
	TotalWalkSteps  int64
	TreesPerSample  int
	AvgStepsPerTree float64
}

// ProfileNativeDense 对 NativeEngine.PredictDense 计时，并统计树遍历步数。
// predictions 长度须 ≥ NOutputGroups()*nrows；计时包含完整 PredictDense 调用。
func ProfileNativeDense(
	e *NativeEngine,
	vals []float64, nrows, ncols int,
	predictions []float64,
	nEstimators int,
) (DenseProfile, error) {
	var prof DenseProfile
	if e == nil || e.forest == nil {
		return prof, nil
	}
	prof.Rows = nrows
	nEst := e.adjustNEstimators(nEstimators)
	prof.TreesPerSample = forestTreeCount(e.forest, nEst)

	start := time.Now()
	err := e.PredictDense(vals, nrows, ncols, predictions, nEstimators)
	prof.Elapsed = time.Since(start)
	if err != nil {
		return prof, err
	}

	for i := 0; i < nrows; i++ {
		fvals := vals[i*ncols : (i+1)*ncols]
		ws := ProfileWalkStats(e.forest, fvals, nEst)
		prof.TotalWalkSteps += int64(ws.Steps)
	}
	if prof.TreesPerSample > 0 && prof.Rows > 0 {
		denom := float64(prof.TreesPerSample) * float64(prof.Rows)
		prof.AvgStepsPerTree = float64(prof.TotalWalkSteps) / denom
	}
	return prof, nil
}

func forestTreeCount(f *ForestIR, nEst int) int {
	if f == nil || nEst <= 0 {
		return 0
	}
	count := 0
	if len(f.IterationIndptr) > 1 {
		for iter := 0; iter < nEst; iter++ {
			if iter+1 >= len(f.IterationIndptr) {
				break
			}
			count += int(f.IterationIndptr[iter+1] - f.IterationIndptr[iter])
		}
		return count
	}
	g := f.NumOutputGroups
	if g <= 0 {
		g = 1
	}
	for i := 0; i < nEst; i++ {
		for k := 0; k < g; k++ {
			if i*g+k < len(f.Trees) {
				count++
			}
		}
	}
	return count
}

func walkTreeSteps(t *TreeIR, fvals []float64) int {
	nodeIdx := int32(0)
	steps := 0
	for {
		steps++
		leftIsLeaf := t.LeftChild[nodeIdx] < 0
		rightIsLeaf := t.RightChild[nodeIdx] < 0
		goLeft := treeDecision(t, int(nodeIdx), fvals)
		if goLeft {
			if leftIsLeaf {
				return steps
			}
			nodeIdx = t.LeftChild[nodeIdx]
		} else {
			if rightIsLeaf {
				return steps
			}
			nodeIdx = t.RightChild[nodeIdx]
		}
	}
}
