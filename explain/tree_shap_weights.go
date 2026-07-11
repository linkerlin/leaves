package explain

import "github.com/linkerlin/leaves/tree"

// computeNodeWeightsArr 返回每个非叶节点的覆盖权重（叶子恒为 1）。
// 优先使用 SumHess；缺失时用子树权重和。LIB-22：替代 map[int32]float64 以降低分配。
func computeNodeWeightsArr(t *tree.TreeIR) []float64 {
	if t == nil || t.NumNodes == 0 {
		return nil
	}
	w := make([]float64, t.NumNodes)
	var fill func(node int32) float64
	fill = func(node int32) float64 {
		if node < 0 {
			return 1.0
		}
		ni := int(node)
		if ni < 0 || ni >= t.NumNodes {
			return 1.0
		}
		if w[ni] > 0 {
			return w[ni]
		}
		if ni < len(t.SumHess) && t.SumHess[ni] > 0 {
			w[ni] = t.SumHess[ni]
			return w[ni]
		}
		lw := fill(t.LeftChild[ni])
		rw := fill(t.RightChild[ni])
		w[ni] = lw + rw
		if w[ni] <= 0 {
			w[ni] = 1.0
		}
		return w[ni]
	}
	fill(0)
	return w
}

func nodeWeightAt(w []float64, node int32) float64 {
	if node < 0 {
		return 1.0
	}
	ni := int(node)
	if ni < 0 || ni >= len(w) || w[ni] <= 0 {
		return 1.0
	}
	return w[ni]
}
