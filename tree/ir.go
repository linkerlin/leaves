// Package tree 定义通用的树模型中间表示 (IR) 和推理引擎接口。
// ForestIR/TreeIR 是来自不同 GBRT 框架（LightGBM、XGBoost、scikit-learn）的
// 模型加载后统一转换的目标格式，供各种推理引擎后端使用。
package tree

// ForestIR 是所有树模型的通用中间表示，与框架来源无关。
// LightGBM / XGBoost / scikit-learn 模型加载后统一转为此格式。
type ForestIR struct {
	// NumFeatures 模型期望的输入特征数。
	NumFeatures int
	// NumOutputGroups 原始输出维度（1 = 回归/二分类，N = 多分类）。
	NumOutputGroups int
	// Trees 树列表。对于多分类模型，树按 [group0_tree0, group1_tree0, ..., group0_tree1, ...] 排列。
	Trees []TreeIR
	// BaseScore 基础分（XGBoost 的 base_score，LightGBM 通常为 0）。
	BaseScore float64
	// WeightDrop 每棵树的权重（XGBoost DART 模型使用，非 DART 则全为 1.0）。
	WeightDrop []float64
	// AverageOutput 是否对树输出取平均（LightGBM RandomForest 模式）。
	AverageOutput bool
	// Name 模型名称（如 "xgboost.gbtree", "lightgbm.gbdt"）。
	Name string

	// ---- XGBoost 3.x 元数据（Phase 0.5+）----

	// TreeInfo 每棵树所属 output group，长度 = len(Trees)。
	TreeInfo []int
	// IterationIndptr boosting 层边界，长度 = num_iterations+1。
	// iteration i 的树为 Trees[IterationIndptr[i]:IterationIndptr[i+1]]。
	IterationIndptr []int
	// NumParallelTree 每轮并行树数（随机森林模式）。
	NumParallelTree int
	// BaseScores 向量 base_score（XGBoost 3.1+）；为空时用 BaseScore 标量。
	BaseScores []float64
}

// NEstimators 返回每组的估计器（树）数量。
func (f *ForestIR) NEstimators() int {
	if f.NumOutputGroups == 0 {
		return 0
	}
	if len(f.IterationIndptr) > 1 {
		return len(f.IterationIndptr) - 1
	}
	return len(f.Trees) / f.NumOutputGroups
}

// TreesForIteration 返回第 layer 轮使用的树数量（含所有 output group）。
func (f *ForestIR) TreesForIteration(layer int) int {
	if len(f.IterationIndptr) < 2 {
		return 0
	}
	if layer < 0 || layer+1 >= len(f.IterationIndptr) {
		return 0
	}
	return f.IterationIndptr[layer+1] - f.IterationIndptr[layer]
}

// TreeCountForNEstimators 返回使用前 nEstimators 轮时的树总数。
func (f *ForestIR) TreeCountForNEstimators(nEstimators int) int {
	if nEstimators <= 0 || nEstimators >= f.NEstimators() {
		return len(f.Trees)
	}
	if len(f.IterationIndptr) > 1 && nEstimators < len(f.IterationIndptr)-1 {
		return f.IterationIndptr[nEstimators]
	}
	return nEstimators * f.NumOutputGroups
}

// NLeaves 返回每棵树的叶子数，长度 = NumOutputGroups * NEstimators()。
func (f *ForestIR) NLeaves() []int {
	n := make([]int, len(f.Trees))
	for i, t := range f.Trees {
		n[i] = t.NLeaves()
	}
	return n
}

// TreeIR 单棵决策树的中间表示。
// 使用完整二叉树布局：节点按深度优先排列，非叶节点存储分裂信息，叶节点用负索引标记。
type TreeIR struct {
	// NumLeaves 叶子节点数量。
	NumLeaves int
	// NumNodes 非叶节点数量 (= NumLeaves - 1，仅对标准二叉树成立)。
	NumNodes int
	// MaxDepth 树的最大深度。
	MaxDepth int

	// SplitFeature [numNodes] 每个非叶节点的分裂特征索引。叶节点此值无意义。
	SplitFeature []int32
	// SplitThreshold [numNodes] 数值分裂阈值。分类节点此值有不同含义。
	SplitThreshold []float64
	// DefaultLeft [numNodes] 缺失值是否默认走左分支。
	DefaultLeft []bool
	// MissingZero [numNodes] 零值视为缺失（走 DefaultLeft）。
	MissingZero []bool
	// MissingNan [numNodes] NaN 视为缺失（走 DefaultLeft）。
	MissingNan []bool

	// LeftChild [numNodes] 左子节点索引。若为叶则为叶子值表中的索引（与 LeafValue 一一对应）。
	LeftChild []int32
	// RightChild [numNodes] 右子节点索引。同上。
	RightChild []int32

	// LeafValue [numLeaves] 叶子的预测值。
	LeafValue []float64

	// ---- 分类特征（LightGBM 特有）----

	// IsCategorical [numNodes] 该节点是否为分类分裂（LightGBM categorical feature）。
	IsCategorical []bool
	// CatOneHot [numNodes] 分类节点是否使用 one-hot 匹配（仅一条值）。
	CatOneHot []bool
	// CatSmall [numNodes] 分类节点是否使用 32 位 bitset（≤32 个值）。
	CatSmall []bool
	// CatBoundaries 分类 bitset 的区间边界，[boundaries[i]:boundaries[i+1]] 为第 i 组。
	CatBoundaries []uint32
	// CatThresholds 分类 bitset 数据。
	CatThresholds []uint32

	// ---- 分裂信息（用于特征重要性计算，Phase 3 使用）----

	// SplitGain [numNodes] 每个节点的分裂增益。XGBoost 提供，LightGBM 可选。
	SplitGain []float64
	// SumHess [numNodes] 每个节点的 Hessian 和（覆盖率）。XGBoost 提供。
	SumHess []float64

	// OutputDim 向量叶维度（XGBoost size_leaf_vector；默认 1）。
	OutputDim int
}

// NLeaves 返回叶子数。
func (t *TreeIR) NLeaves() int {
	return t.NumLeaves
}

// IsTreeEmpty 判断树是否为空（无叶子）。
func (t *TreeIR) IsTreeEmpty() bool {
	return t.NumLeaves == 0
}
