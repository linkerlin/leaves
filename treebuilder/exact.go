package treebuilder

import (
	"sort"

	"github.com/dmitryikh/leaves/data"
	"github.com/dmitryikh/leaves/tree"
)

// Config 树构建超参。
type Config struct {
	MaxDepth       int
	MinHessian     float64
	Lambda         float64
	Gamma          float64
	LearningRate   float64
	MaxBin         int
	FeatureIndices []int
	NumThreads     int  // 0 = runtime.NumCPU()；T4 多线程 hist
	UseGPUHist     bool // 内部：gpu_hist 时启用 GoMLX 直方图加速
}

func featureList(cfg Config, ncols int) []int {
	if len(cfg.FeatureIndices) > 0 {
		return cfg.FeatureIndices
	}
	out := make([]int, ncols)
	for i := range out {
		out[i] = i
	}
	return out
}

// BuildExact 精确贪心建树（小数据 MVP）。
func BuildExact(dm data.Matrix, indices []int, grad, hess []float64, cfg Config) *tree.TreeIR {
	if cfg.MaxDepth <= 0 {
		cfg.MaxDepth = 6
	}
	if cfg.MinHessian <= 0 {
		cfg.MinHessian = 1e-3
	}
	if cfg.Lambda < 0 {
		cfg.Lambda = 1.0
	}
	if cfg.LearningRate <= 0 {
		cfg.LearningRate = 0.3
	}
	root := buildNode(dm, indices, grad, hess, 0, cfg)
	if root == nil {
		w := leafWeight(indices, grad, hess, cfg.Lambda) * cfg.LearningRate
		return tree.BuildTreeIR(nil, []float64{w}, nil, nil, 0)
	}
	nodes, leaves := flatten(root)
	return tree.BuildTreeIR(nodes, leaves, nil, nil, 0)
}

type node struct {
	feat      int
	threshold float64
	left      *node
	right     *node
	leaf      bool
	leafVal   float64
	sumHess   float64
	catSmall  bool
}

func buildNode(dm data.Matrix, idx []int, grad, hess []float64, depth int, cfg Config) *node {
	sumG, sumH := sumGradHess(idx, grad, hess)
	if sumH < cfg.MinHessian || depth >= cfg.MaxDepth || len(idx) <= 1 {
		return &node{
			leaf:    true,
			leafVal: leafWeightFromSums(sumG, sumH, cfg.Lambda) * cfg.LearningRate,
			sumHess: sumH,
		}
	}

	bestGain := cfg.Gamma
	var bestFeat int
	var bestThr float64
	var bestLeft, bestRight []int
	var bestCat bool

	ncols := dm.NumCol()
	row := make([]float64, ncols)
	for _, f := range featureList(cfg, ncols) {
		if data.IsCategorical(dm, f) {
			gain, thr, left, right, ok := bestCategoricalSplit(dm, idx, f, grad, hess, sumG, sumH, row, cfg)
			if ok && gain > bestGain {
				bestGain = gain
				bestFeat = f
				bestThr = thr
				bestLeft = left
				bestRight = right
				bestCat = true
			}
			continue
		}
		type pair struct {
			val float64
			i   int
		}
		pairs := make([]pair, 0, len(idx))
		for _, i := range idx {
			_ = dm.Row(i, row)
			pairs = append(pairs, pair{row[f], i})
		}
		sort.Slice(pairs, func(a, b int) bool { return pairs[a].val < pairs[b].val })

		for pi := 0; pi < len(pairs)-1; pi++ {
			if pairs[pi].val == pairs[pi+1].val {
				continue
			}
			thr := (pairs[pi].val + pairs[pi+1].val) * 0.5
			left, right := splitIndices(dm, idx, f, thr, row)
			if len(left) == 0 || len(right) == 0 {
				continue
			}
			gl, hl := sumGradHess(left, grad, hess)
			gr, hr := sumGradHess(right, grad, hess)
			gain := splitGain(gl, hl, gr, hr, sumG, sumH, cfg.Lambda)
			if gain > bestGain {
				bestGain = gain
				bestFeat = f
				bestThr = thr
				bestLeft = left
				bestRight = right
				bestCat = false
			}
		}
	}

	if bestGain <= cfg.Gamma {
		return &node{
			leaf:    true,
			leafVal: leafWeightFromSums(sumG, sumH, cfg.Lambda) * cfg.LearningRate,
			sumHess: sumH,
		}
	}

	return &node{
		feat:      bestFeat,
		threshold: bestThr,
		left:      buildNode(dm, bestLeft, grad, hess, depth+1, cfg),
		right:     buildNode(dm, bestRight, grad, hess, depth+1, cfg),
		sumHess:   sumH,
		catSmall:  bestCat,
	}
}

func splitIndices(dm data.Matrix, idx []int, feat int, thr float64, row []float64) (left, right []int) {
	for _, i := range idx {
		_ = dm.Row(i, row)
		if row[feat] <= thr {
			left = append(left, i)
		} else {
			right = append(right, i)
		}
	}
	return left, right
}

func sumGradHess(idx []int, grad, hess []float64) (g, h float64) {
	for _, i := range idx {
		g += grad[i]
		h += hess[i]
	}
	return g, h
}

func leafWeight(idx []int, grad, hess []float64, lambda float64) float64 {
	g, h := sumGradHess(idx, grad, hess)
	return leafWeightFromSums(g, h, lambda)
}

func leafWeightFromSums(g, h, lambda float64) float64 {
	return -g / (h + lambda)
}

func splitGain(gl, hl, gr, hr, g, h, lambda float64) float64 {
	if hl <= 0 || hr <= 0 {
		return 0
	}
	left := gl * gl / (hl + lambda)
	right := gr * gr / (hr + lambda)
	total := g * g / (h + lambda)
	return 0.5 * (left + right - total)
}

func flatten(n *node) ([]tree.LgNodeData, []float64) {
	if n == nil || n.leaf {
		val := 0.0
		if n != nil {
			val = n.leafVal
		}
		return nil, []float64{val}
	}
	var countInternal func(*node) int
	countInternal = func(cur *node) int {
		if cur.leaf {
			return 0
		}
		return 1 + countInternal(cur.left) + countInternal(cur.right)
	}
	nodes := make([]tree.LgNodeData, countInternal(n))
	var leaves []float64
	var nextInternal uint32
	var fill func(*node) uint32
	fill = func(cur *node) uint32 {
		if cur.leaf {
			idx := uint32(len(leaves))
			leaves = append(leaves, cur.leafVal)
			return idx
		}
		myIdx := nextInternal
		nextInternal++
		leftIdx := fill(cur.left)
		rightIdx := fill(cur.right)
		nd := tree.LgNodeData{
			Feature:   uint32(cur.feat),
			Threshold: cur.threshold,
			Flags:     flagMissingNan,
		}
		if cur.catSmall {
			nd.Flags |= flagCategorical | flagCatSmall
			nd.Threshold = float64(uint32(1) << uint32(int(cur.threshold)))
		}
		if cur.left.leaf {
			nd.Flags |= flagLeftLeaf
		}
		nd.Left = leftIdx
		if cur.right.leaf {
			nd.Flags |= flagRightLeaf
		}
		nd.Right = rightIdx
		nodes[myIdx] = nd
		return myIdx
	}
	fill(n)
	return nodes, leaves
}

const (
	flagMissingNan    = 1 << 5
	flagLeftLeaf      = 1 << 2
	flagRightLeaf     = 1 << 3
	flagCategorical   = 1 << 0
	flagCatSmall      = 1 << 7
)
