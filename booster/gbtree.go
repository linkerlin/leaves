package booster

import (
	"math/rand"

	"github.com/dmitryikh/leaves/data"
	"github.com/dmitryikh/leaves/tree"
	"github.com/dmitryikh/leaves/treebuilder"
)

// TrainParams 每轮训练采样参数。
type TrainParams struct {
	Subsample       float64
	ColsampleByTree float64
	Seed            int64
	NumParallelTree int
	DART            *DARTConfig
}

// GBTree 梯度提升树 booster。
type GBTree struct {
	forest     *tree.ForestIR
	cfg        treebuilder.Config
	treeMethod string
	train      TrainParams
	rng        *rand.Rand
}

// NewGBTree 创建空 booster。
func NewGBTree(numFeatures int, baseScore float64, numOutputs int, cfg treebuilder.Config, treeMethod string, train TrainParams) *GBTree {
	if treeMethod == "" {
		treeMethod = treebuilder.MethodHist
	}
	if numOutputs <= 0 {
		numOutputs = 1
	}
	if train.NumParallelTree <= 0 {
		train.NumParallelTree = 1
	}
	rng := rand.New(rand.NewSource(train.Seed))
	f := &tree.ForestIR{
		NumFeatures:       numFeatures,
		NumOutputGroups:   numOutputs,
		BaseScore:         baseScore,
		Name:              "leaves.gbtree",
		IterationIndptr:   []int{0},
		TreeInfo:          nil,
		Trees:             nil,
		WeightDrop:        nil,
	}
	if train.NumParallelTree > 1 {
		f.NumParallelTree = train.NumParallelTree
		f.AverageOutput = true
		f.Name = "leaves.rf"
	}
	if train.DART != nil && train.DART.RateDrop > 0 {
		f.Name = "leaves.dart"
	}
	return &GBTree{
		cfg:        cfg,
		treeMethod: treeMethod,
		train:      train,
		rng:        rng,
		forest:     f,
	}
}

func (b *GBTree) Forest() *tree.ForestIR { return b.forest }

// NewGBTreeFromForest 从已有 ForestIR 恢复 booster（checkpoint 续训）。
func NewGBTreeFromForest(f *tree.ForestIR, cfg treebuilder.Config, treeMethod string, train TrainParams) *GBTree {
	if train.NumParallelTree <= 0 {
		train.NumParallelTree = 1
	}
	rng := rand.New(rand.NewSource(train.Seed))
	return &GBTree{
		cfg:        cfg,
		treeMethod: treeMethod,
		train:      train,
		rng:        rng,
		forest:     f,
	}
}

func (b *GBTree) setLearningRate(lr float64) { b.cfg.LearningRate = lr }

func (b *GBTree) NumOutputGroups() int { return b.forest.NumOutputGroups }

// Boost 用当前梯度/Hessian 建树并加入森林。
func (b *GBTree) Boost(dm data.Matrix, grad, hess []float64) {
	n := dm.NumRow()
	g := b.forest.NumOutputGroups
	par := b.train.NumParallelTree
	beforeTrees := len(b.forest.Trees)

	for p := 0; p < par; p++ {
		idx := data.SubsampleIndices(n, b.train.Subsample, b.rng)
		featIdx := data.ColsampleIndices(dm.NumCol(), b.train.ColsampleByTree, b.rng)
		cfg := b.cfg
		cfg.FeatureIndices = featIdx

		if g <= 1 {
			tir := treebuilder.Build(dm, idx, grad, hess, cfg, b.treeMethod)
			b.appendTree(*tir, 0)
			continue
		}
		for k := 0; k < g; k++ {
			gk := make([]float64, n)
			hk := make([]float64, n)
			for i := 0; i < n; i++ {
				gk[i], hk[i] = gradHessAt(grad, hess, i, k, g)
			}
			tir := treebuilder.Build(dm, idx, gk, hk, cfg, b.treeMethod)
			b.appendTree(*tir, k)
		}
	}

	numNew := len(b.forest.Trees) - beforeTrees
	if b.train.DART != nil && b.train.DART.RateDrop > 0 {
		ApplyDARTDrop(b.forest, *b.train.DART, numNew, b.rng)
	}
	b.forest.IterationIndptr = append(b.forest.IterationIndptr, len(b.forest.Trees))
}

func (b *GBTree) appendTree(tir tree.TreeIR, classIdx int) {
	b.forest.Trees = append(b.forest.Trees, tir)
	b.forest.WeightDrop = append(b.forest.WeightDrop, 1.0)
	b.forest.TreeInfo = append(b.forest.TreeInfo, classIdx)
}

// PredictMargins 批量 raw margin。
func (b *GBTree) PredictMargins(dm data.Matrix, out []float64) {
	n := dm.NumRow()
	g := b.forest.NumOutputGroups
	row := make([]float64, dm.NumCol())
	for i := 0; i < n; i++ {
		_ = dm.Row(i, row)
		margins := tree.ForestMargins(b.forest, row, 0)
		base := i * g
		for k := 0; k < g && k < len(margins); k++ {
			if base+k < len(out) {
				out[base+k] = margins[k]
			}
		}
	}
}
