package rankutil

import (
	"fmt"
	"sort"

	"github.com/dmitryikh/leaves/data"
)

// RankedItem 单条候选（组内一行）。
type RankedItem struct {
	RowInGroup int
	Label      float64
	Score      float64
}

// GroupSlice 取出第 groupIdx 个 query 的特征行、标签与全局行偏移。
func GroupSlice(dm *data.DenseWithGroups, groupIdx int) (start, count int, err error) {
	if dm == nil {
		return 0, 0, fmt.Errorf("nil matrix")
	}
	g := dm.Groups()
	if groupIdx < 0 || groupIdx >= len(g) {
		return 0, 0, fmt.Errorf("group %d out of range [0,%d)", groupIdx, len(g))
	}
	start = 0
	for i := 0; i < groupIdx; i++ {
		start += g[i]
	}
	return start, g[groupIdx], nil
}

// RankGroup 对组内样本按预测分降序排列。
func RankGroup(dm *data.DenseWithGroups, preds []float64, groupIdx int, topK int) ([]RankedItem, error) {
	start, count, err := GroupSlice(dm, groupIdx)
	if err != nil {
		return nil, err
	}
	labels := dm.Labels()
	items := make([]RankedItem, count)
	for i := 0; i < count; i++ {
		row := start + i
		items[i] = RankedItem{
			RowInGroup: i,
			Label:      labels[row],
			Score:      preds[row],
		}
	}
	sort.Slice(items, func(i, j int) bool {
		return items[i].Score > items[j].Score
	})
	if topK > 0 && topK < len(items) {
		items = items[:topK]
	}
	return items, nil
}

// GroupQID 返回组在 TSV 中的 qid（等于训练集内顺序下标；测试集从 train_users 起）。
func GroupQID(groupIdx int, trainUserCount int) int {
	return trainUserCount + groupIdx
}
