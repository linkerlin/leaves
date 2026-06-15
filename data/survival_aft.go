package data

import (
	"fmt"
	"math"
)

// AFTInterval survival:aft 区间删失标签（对标 XGBoost lower/upper bound）。
//   - 事件观测：Lower == Upper == t
//   - 右删失于 t：Lower = t，Upper = +Inf
//   - 左删失于 t：Lower = 0，Upper = t
//   - 区间删失：(Lower, Upper)
type AFTInterval struct {
	Lower float64
	Upper float64
}

// AFTIntervalMatrix 带区间删失标签的训练矩阵。
type AFTIntervalMatrix interface {
	Matrix
	AFTIntervals() []AFTInterval
}

// AFTDense 稠密特征 + 区间删失标签。
type AFTDense struct {
	*Dense
	Intervals []AFTInterval
}

// NewAFTDense 创建 AFT 区间删失矩阵。
func NewAFTDense(d *Dense, intervals []AFTInterval) (*AFTDense, error) {
	if d == nil {
		return nil, fmt.Errorf("data: nil dense")
	}
	if len(intervals) != d.Rows {
		return nil, fmt.Errorf("data: intervals %d != rows %d", len(intervals), d.Rows)
	}
	for i, iv := range intervals {
		if err := iv.Validate(); err != nil {
			return nil, fmt.Errorf("data: row %d: %w", i, err)
		}
	}
	return &AFTDense{Dense: d, Intervals: intervals}, nil
}

func (a *AFTDense) AFTIntervals() []AFTInterval { return a.Intervals }

// AFTIntervalsOf 若 dm 实现 AFTIntervalMatrix 则返回区间标签。
func AFTIntervalsOf(dm Matrix) ([]AFTInterval, bool) {
	im, ok := dm.(AFTIntervalMatrix)
	if !ok {
		return nil, false
	}
	return im.AFTIntervals(), true
}

// Validate 校验区间语义。
func (iv AFTInterval) Validate() error {
	if iv.Lower < 0 {
		return fmt.Errorf("lower %v must be >= 0", iv.Lower)
	}
	if math.IsInf(iv.Upper, 1) {
		if iv.Lower <= 0 {
			return fmt.Errorf("right censor needs lower > 0")
		}
		return nil
	}
	if iv.Upper < iv.Lower {
		return fmt.Errorf("upper %v must be >= lower %v", iv.Upper, iv.Lower)
	}
	if iv.Upper == iv.Lower && iv.Lower <= 0 {
		return fmt.Errorf("exact event needs time > 0")
	}
	return nil
}

// AFTIntervalFromScalarLabel Cox 式标量标签 → 区间（正=事件，负=-右删失时间）。
func AFTIntervalFromScalarLabel(y float64) AFTInterval {
	if y > 0 {
		return AFTInterval{Lower: y, Upper: y}
	}
	if y < 0 {
		return AFTInterval{Lower: -y, Upper: math.Inf(1)}
	}
	return AFTInterval{Lower: 0, Upper: 0}
}

// ScalarLabelsFromAFTIntervals 将区间编码为 Dense.Labels 占位（下界；右删失为负）。
func ScalarLabelsFromAFTIntervals(ivs []AFTInterval) []float64 {
	out := make([]float64, len(ivs))
	for i, iv := range ivs {
		if math.IsInf(iv.Upper, 1) {
			out[i] = -iv.Lower
		} else if iv.Lower == iv.Upper {
			out[i] = iv.Lower
		} else {
			out[i] = iv.Lower
		}
	}
	return out
}
