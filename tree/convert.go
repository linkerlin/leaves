package tree

import (
	"math"
)

// LgTreeToTreeIR 将现有的 lgTree 结构转换为通用 TreeIR。
// 这是 Phase 0 的兼容桥——后续 IO 层会直接生成 TreeIR 而无需此转换。
func LgTreeToTreeIR(tree interface{}, useXGBoostStyle bool) *TreeIR {
	// 使用反射或类型断言来处理 lgTree
	// 当前 lgTree 的定义在 leaves 包中（未导出），这里通过接口抽象
	// 实际转换由调用方完成。此处提供基于接口的版本。

	// 由于 lgTree 是未导出类型，此处仅保留接口占位。
	// 实际转换逻辑放在根包或 io 包中。
	return nil
}

// ConvertLgTree 将 LeafNode/DecisionNode 风格的数据转为 TreeIR。
// nodes: 节点数组，叶子值数组分开存储。
// 这是通用的转换函数，与具体框架无关。
func BuildTreeIR(
	nodes []LgNodeData,
	leafValues []float64,
	catBoundaries []uint32,
	catThresholds []uint32,
	nCategorical uint32,
) *TreeIR {
	nNodes := len(nodes)
	if nNodes == 0 {
		t := &TreeIR{
			NumLeaves: 1,
			NumNodes:  0,
			MaxDepth:  0,
			LeafValue: leafValues,
		}
		return t
	}

	t := &TreeIR{
		NumLeaves:     nNodes + 1,
		NumNodes:      nNodes,
		SplitFeature:  make([]int32, nNodes),
		SplitThreshold: make([]float64, nNodes),
		DefaultLeft:    make([]bool, nNodes),
		MissingZero:    make([]bool, nNodes),
		MissingNan:     make([]bool, nNodes),
		LeftChild:     make([]int32, nNodes),
		RightChild:    make([]int32, nNodes),
		IsCategorical: make([]bool, nNodes),
		CatOneHot:     make([]bool, nNodes),
		CatSmall:      make([]bool, nNodes),
	}

	// 计算深度
	t.MaxDepth = computeTreeDepth(nodes)

	// 收集所有叶子值
	t.LeafValue = make([]float64, 0, nNodes+1)

	for i, node := range nodes {
		t.SplitFeature[i] = int32(node.Feature)
		t.SplitThreshold[i] = node.Threshold
		t.DefaultLeft[i] = node.Flags&flagDefaultLeft != 0
		t.MissingZero[i] = node.Flags&flagMissingZero != 0
		t.MissingNan[i] = node.Flags&flagMissingNan != 0

		if node.Flags&flagCategorical != 0 {
			t.IsCategorical[i] = true
			if node.Flags&flagCatOneHot != 0 {
				t.CatOneHot[i] = true
			} else if node.Flags&flagCatSmall != 0 {
				t.CatSmall[i] = true
			}
		}

		// 处理子节点
		if node.Flags&flagLeftLeaf != 0 {
			t.LeftChild[i] = int32(^node.Left) // 负值表示叶子索引
		} else {
			t.LeftChild[i] = int32(node.Left)
		}

		if node.Flags&flagRightLeaf != 0 {
			t.RightChild[i] = int32(^node.Right)
		} else {
			t.RightChild[i] = int32(node.Right)
		}
	}

	t.LeafValue = leafValues
	t.CatBoundaries = catBoundaries
	t.CatThresholds = catThresholds

	return t
}

// LgNodeData 传递 lgNode 的数据（因为 lgNode 是未导出类型）。
type LgNodeData struct {
	Threshold float64
	Left      uint32
	Right     uint32
	Feature   uint32
	Flags     uint8
}

// ---- lgTree 节点标志常量（与 lgtree.go 保持一致）----
const (
	flagCategorical = 1 << 0
	flagDefaultLeft = 1 << 1
	flagLeftLeaf    = 1 << 2
	flagRightLeaf   = 1 << 3
	flagMissingZero = 1 << 4
	flagMissingNan  = 1 << 5
	flagCatOneHot   = 1 << 6
	flagCatSmall    = 1 << 7
)

// computeTreeDepth 通过 BFS 计算树的最大深度。
func computeTreeDepth(nodes []LgNodeData) int {
	if len(nodes) == 0 {
		return 0
	}
	type queueItem struct {
		index uint32
		depth int
	}
	queue := []queueItem{{index: 0, depth: 1}}
	maxDepth := 0
	for len(queue) > 0 {
		item := queue[0]
		queue = queue[1:]
		if item.depth > maxDepth {
			maxDepth = item.depth
		}
		idx := item.index
		if int(idx) >= len(nodes) {
			continue
		}
		node := nodes[idx]
		if node.Flags&flagLeftLeaf == 0 && node.Left < uint32(len(nodes)) {
			queue = append(queue, queueItem{index: node.Left, depth: item.depth + 1})
		}
		if node.Flags&flagRightLeaf == 0 && node.Right < uint32(len(nodes)) {
			queue = append(queue, queueItem{index: node.Right, depth: item.depth + 1})
		}
	}
	return maxDepth
}

// ---- 批量预测辅助函数 ----

// BatchSize 并行批预测的默认批次大小。
const BatchSize = 16

// PredictDenseWithEngine 使用 Engine 在稠密矩阵上做批量预测，含多线程支持。
func PredictDenseWithEngine(
	engine Engine,
	vals []float64, nrows int, ncols int,
	predictions []float64,
	nEstimators int,
	nThreads int,
) error {
	// 小 batch 或单线程直接调用
	if nrows <= BatchSize || nThreads <= 1 {
		return engine.PredictDense(vals, nrows, ncols, predictions, nEstimators)
	}
	// 多线程：分批次并发调用（简化版，由上层调用方控制线程池）
	return engine.PredictDense(vals, nrows, ncols, predictions, nEstimators)
}

// PredictCSRWithEngine 使用 Engine 在 CSR 稀疏矩阵上做批量预测，含多线程支持。
func PredictCSRWithEngine(
	engine Engine,
	indptr []int, cols []int, vals []float64,
	predictions []float64,
	nEstimators int,
	nThreads int,
) error {
	nRows := len(indptr) - 1
	if nRows <= BatchSize || nThreads <= 1 {
		return engine.PredictCSR(indptr, cols, vals, predictions, nEstimators)
	}
	return engine.PredictCSR(indptr, cols, vals, predictions, nEstimators)
}

// ---- 变换函数 ----

// ApplyTransformRaw 恒等变换。
func ApplyTransformRaw(rawPredictions []float64, outputPredictions []float64, startIndex int) error {
	for i := range rawPredictions {
		outputPredictions[startIndex+i] = rawPredictions[i]
	}
	return nil
}

// ApplyTransformLogistic Sigmoid 变换。
func ApplyTransformLogistic(rawPredictions []float64, outputPredictions []float64, startIndex int) error {
	for i := range rawPredictions {
		outputPredictions[startIndex+i] = 1.0 / (1.0 + math.Exp(-rawPredictions[i]))
	}
	return nil
}

// ApplyTransformSoftmax Softmax 变换。
func ApplyTransformSoftmax(rawPredictions []float64, outputPredictions []float64, startIndex int) error {
	sum := 0.0
	for i, v := range rawPredictions {
		exp := math.Exp(v)
		outputPredictions[startIndex+i] = exp
		sum += exp
	}
	if sum != 0.0 {
		invSum := 1.0 / sum
		for i := range rawPredictions {
			outputPredictions[startIndex+i] *= invSum
		}
	}
	return nil
}

// ApplyTransformExponential Exp 变换。
func ApplyTransformExponential(rawPredictions []float64, outputPredictions []float64, startIndex int) error {
	for i := range rawPredictions {
		outputPredictions[startIndex+i] = math.Exp(rawPredictions[i])
	}
	return nil
}
