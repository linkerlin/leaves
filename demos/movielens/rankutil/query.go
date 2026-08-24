package rankutil

import (
	"github.com/linkerlin/leaves/v2/data"
	recrank "github.com/linkerlin/leaves/v2/recsys/rankutil"
)

// RankedItem 组内候选（与 recsys/rankutil 同形）。
type RankedItem = recrank.RankedItem

// GroupSlice 委托 recsys/rankutil。
func GroupSlice(dm *data.DenseWithGroups, groupIdx int) (start, count int, err error) {
	return recrank.GroupSlice(dm, groupIdx)
}

// RankGroup 委托 recsys/rankutil。
func RankGroup(dm *data.DenseWithGroups, preds []float64, groupIdx int, topK int) ([]RankedItem, error) {
	return recrank.RankGroup(dm, preds, groupIdx, topK)
}

// GroupQID 委托 recsys/rankutil。
func GroupQID(groupIdx int, trainUserCount int) int {
	return recrank.GroupQID(groupIdx, trainUserCount)
}
