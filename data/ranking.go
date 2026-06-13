package data

import "fmt"

// RankingMatrix 排序学习数据：样本按 query 分组。
type RankingMatrix interface {
	GroupedMatrix
}

// GroupsFromRanking 校验 groups 并返回副本。
func GroupsFromRanking(dm Matrix) ([]int, error) {
	gm, ok := dm.(GroupedMatrix)
	if !ok {
		return nil, fmt.Errorf("data: matrix has no groups")
	}
	g := gm.Groups()
	if len(g) == 0 {
		return nil, fmt.Errorf("data: empty groups")
	}
	out := make([]int, len(g))
	copy(out, g)
	return out, nil
}

// DenseWithGroups 在 Dense 上附加 group 信息。
type DenseWithGroups struct {
	*Dense
	GroupSizes []int
}

func (d *DenseWithGroups) Groups() []int { return d.GroupSizes }

// NewDenseWithGroups 创建带 groups 的 Dense。
func NewDenseWithGroups(d *Dense, groups []int) (*DenseWithGroups, error) {
	if d == nil {
		return nil, fmt.Errorf("data: nil dense")
	}
	sum := 0
	for _, g := range groups {
		sum += g
	}
	if sum != d.Rows {
		return nil, fmt.Errorf("data: groups sum %d != rows %d", sum, d.Rows)
	}
	return &DenseWithGroups{Dense: d, GroupSizes: groups}, nil
}
