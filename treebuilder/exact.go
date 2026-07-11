package treebuilder

import (
	"sort"

	"github.com/linkerlin/leaves/data"
	"github.com/linkerlin/leaves/tree"
)

// Config 树构建超参。
type Config struct {
	MaxDepth            int
	MaxLeaves           int // lossguide：单棵树最大叶子数；0=不限
	MinHessian          float64
	Lambda              float64
	Gamma               float64
	LearningRate        float64
	MaxBin              int
	FeatureIndices      []int
	NumThreads          int    // 0 = runtime.NumCPU()；T4 多线程 hist
	UseGPUHist          bool   // hist/gpu_hist：尝试 WebGPU 增益扫描，失败回退 Born CPU / 纯 CPU
	AccelMode           string // auto|webgpu|born_cpu|cpu；空则读 LEAVES_TRAIN_ACCEL
	HistBinPolicy       string // global（默认）| per_node
	GlobalBins          *GlobalHistBins
	MonotoneConstraints []int // 每特征 -1/0/1；长度可小于列数，不足视为 0
	OutputDim           int   // 向量叶维度 k（multi_output_tree）；0 或 1 = 标量叶
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
	k := cfg.OutputDim
	if k < 1 {
		k = 1
	}
	root := buildNode(dm, indices, grad, hess, 0, cfg, intPtr1())
	if root == nil {
		w := leafWeight(indices, grad, hess, k, cfg.Lambda, cfg.LearningRate)
		t := tree.BuildTreeIR(nil, w, nil, nil, 0)
		t.OutputDim = k
		return t
	}
	nodes, leaves := flatten(root)
	t := tree.BuildTreeIR(nodes, leaves, nil, nil, 0)
	t.OutputDim = k
	return t
}

type node struct {
	feat      int
	threshold float64
	left      *node
	right     *node
	leaf      bool
	leafVal   []float64 // 长度 k；标量叶 k=1
	sumHess   float64   // 总 hessian（跨类求和），用于停止/覆盖判断
	catSmall  bool
}

func buildNode(dm data.Matrix, idx []int, grad, hess []float64, depth int, cfg Config, leaves *int) *node {
	k := cfg.OutputDim
	if k < 1 {
		k = 1
	}
	sumG, sumH := sumGradHess(idx, grad, hess, k)
	totalH := 0.0
	for c := 0; c < k; c++ {
		totalH += sumH[c]
	}
	if totalH < cfg.MinHessian || depth >= cfg.MaxDepth || len(idx) <= 1 || leafBudgetExceeded(cfg, leaves) {
		return &node{
			leaf:    true,
			leafVal: leafWeightFromSums(sumG, sumH, cfg.Lambda, cfg.LearningRate),
			sumHess: totalH,
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
			gain, thr, left, right, ok := bestCategoricalSplit(dm, idx, f, grad, hess, sumG, sumH, k, row, cfg)
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
			gl, hl := sumGradHess(left, grad, hess, k)
			gr, hr := sumGradHess(right, grad, hess, k)
			gain := splitGain(gl, hl, gr, hr, sumG, sumH, cfg.Lambda)
			if gain > bestGain && monotoneAllowsSplit(cfg, f, left, right, grad, hess, k) {
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
			leafVal: leafWeightFromSums(sumG, sumH, cfg.Lambda, cfg.LearningRate),
			sumHess: totalH,
		}
	}

	return &node{
		feat:      bestFeat,
		threshold: bestThr,
		left:      buildNode(dm, bestLeft, grad, hess, depth+1, cfg, splitBudget(cfg, leaves)),
		right:     buildNode(dm, bestRight, grad, hess, depth+1, cfg, leaves),
		sumHess:   totalH,
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

// sumGradHess 累加 idx 中各样本的逐类 grad/hess。grad/hess 布局为 [n*k] 行主序
// （grad[i*k+c]）；k=1 退化为标量（返回 len-1 切片）。
func sumGradHess(idx []int, grad, hess []float64, k int) (sumG, sumH []float64) {
	sumG = make([]float64, k)
	sumH = make([]float64, k)
	for _, i := range idx {
		base := i * k
		for c := 0; c < k; c++ {
			sumG[c] += grad[base+c]
			sumH[c] += hess[base+c]
		}
	}
	return sumG, sumH
}

// leafWeightFromSums 逐类叶权重 -g_c/(h_c+lambda)（已乘学习率）。
func leafWeightFromSums(sumG, sumH []float64, lambda, lr float64) []float64 {
	w := make([]float64, len(sumG))
	for c := range sumG {
		w[c] = -sumG[c] / (sumH[c] + lambda) * lr
	}
	return w
}

func leafWeight(idx []int, grad, hess []float64, k int, lambda, lr float64) []float64 {
	sg, sh := sumGradHess(idx, grad, hess, k)
	return leafWeightFromSums(sg, sh, lambda, lr)
}

// splitGain 多输出分裂增益：同一 (feat,thr) 下逐类增益求和。k=1 即经典标量公式。
// 某类一侧 hessian<=0 时该类贡献 0（不整体否决分裂）。
func splitGain(gl, hl, gr, hr, sumG, sumH []float64, lambda float64) float64 {
	var gain float64
	for c := range sumG {
		if hl[c] <= 0 || hr[c] <= 0 {
			continue
		}
		left := gl[c] * gl[c] / (hl[c] + lambda)
		right := gr[c] * gr[c] / (hr[c] + lambda)
		total := sumG[c] * sumG[c] / (sumH[c] + lambda)
		gain += 0.5 * (left + right - total)
	}
	return gain
}

func flatten(n *node) ([]tree.LgNodeData, []float64) {
	if n == nil {
		return nil, nil
	}
	if n.leaf {
		return nil, append([]float64(nil), n.leafVal...)
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
	var leafCount uint32 // 叶序号（存储为叶引用）；与 leaves 的扁平偏移解耦
	var nextInternal uint32
	var fill func(*node) uint32
	fill = func(cur *node) uint32 {
		if cur.leaf {
			idx := leafCount
			leafCount++
			leaves = append(leaves, cur.leafVal...)
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
	flagMissingNan  = 1 << 5
	flagLeftLeaf    = 1 << 2
	flagRightLeaf   = 1 << 3
	flagCategorical = 1 << 0
	flagCatSmall    = 1 << 7
)
