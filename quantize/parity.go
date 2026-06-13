package quantize

import (
	"fmt"
	"math"

	"github.com/dmitryikh/leaves/tree"
)

// Gate parity 门禁阈值。
type Gate struct {
	// MaxMarginDiff 单样本 margin 最大允许偏差；0 表示仅要求完全一致（量化后通常不可达）。
	MaxMarginDiff float64
	// MaxThresholdErr 数值阈值反量化最大误差上界；0 表示跳过此项。
	MaxThresholdErr float64
	// MaxFailRate 超过 MaxMarginDiff 的样本占比上界；0 表示不允许失败。
	MaxFailRate float64
}

// DefaultGate 默认门禁（smoke 级模型经验值）。
func DefaultGate() Gate {
	return Gate{
		MaxMarginDiff:   0.15,
		MaxThresholdErr: 0,
		MaxFailRate:     0.02,
	}
}

// ParityResult Native vs 量化 margin 对比结果。
type ParityResult struct {
	Samples         int
	MaxMarginDiff   float64
	MeanMarginDiff  float64
	MaxThresholdErr float64
	Failures        int
}

// Pass 是否通过门禁。
func (r ParityResult) Pass(g Gate) bool {
	if r.Samples == 0 {
		return false
	}
	if g.MaxThresholdErr > 0 && r.MaxThresholdErr > g.MaxThresholdErr {
		return false
	}
	if g.MaxMarginDiff > 0 && r.MaxMarginDiff > g.MaxMarginDiff {
		return false
	}
	if g.MaxFailRate >= 0 && g.MaxMarginDiff > 0 {
		rate := float64(r.Failures) / float64(r.Samples)
		if rate > g.MaxFailRate {
			return false
		}
	}
	return true
}

// CheckParity 对比原始 Forest 与量化森林在样本上的 margin 差异。
func CheckParity(orig *tree.ForestIR, qf *QuantizedForest, rows [][]float64, nEstimators int) ParityResult {
	var r ParityResult
	if orig == nil || qf == nil {
		return r
	}
	r.MaxThresholdErr = qf.MaxThresholdQuantError()
	g := orig.NumOutputGroups
	if g <= 0 {
		g = 1
	}
	var sumDiff float64
	for _, fvals := range rows {
		if len(fvals) < orig.NumFeatures {
			continue
		}
		r.Samples++
		native := tree.ForestMargins(orig, fvals, nEstimators)
		quant := forestMarginsQ(qf, fvals, nEstimators)
		var rowMax float64
		for k := 0; k < g; k++ {
			var a, b float64
			if k < len(native) {
				a = native[k]
			}
			if k < len(quant) {
				b = quant[k]
			}
			d := math.Abs(a - b)
			if d > rowMax {
				rowMax = d
			}
		}
		sumDiff += rowMax
		if rowMax > r.MaxMarginDiff {
			r.MaxMarginDiff = rowMax
		}
	}
	if r.Samples > 0 {
		r.MeanMarginDiff = sumDiff / float64(r.Samples)
	}
	return r
}

// CheckParityWithGate 运行 parity 并在未通过时返回 error。
func CheckParityWithGate(orig *tree.ForestIR, qf *QuantizedForest, rows [][]float64, nEstimators int, g Gate) (ParityResult, error) {
	r := CheckParity(orig, qf, rows, nEstimators)
	if g.MaxMarginDiff > 0 {
		for _, fvals := range rows {
			if len(fvals) < orig.NumFeatures {
				continue
			}
			native := tree.ForestMargins(orig, fvals, nEstimators)
			quant := forestMarginsQ(qf, fvals, nEstimators)
			ng := orig.NumOutputGroups
			if ng <= 0 {
				ng = 1
			}
			var rowMax float64
			for k := 0; k < ng; k++ {
				var a, b float64
				if k < len(native) {
					a = native[k]
				}
				if k < len(quant) {
					b = quant[k]
				}
				d := math.Abs(a - b)
				if d > rowMax {
					rowMax = d
				}
			}
			if rowMax > g.MaxMarginDiff {
				r.Failures++
			}
		}
	}
	if !r.Pass(g) {
		return r, fmt.Errorf("quantize: parity gate failed: max_margin_diff=%.6g mean=%.6g max_thresh_err=%.6g failures=%d/%d",
			r.MaxMarginDiff, r.MeanMarginDiff, r.MaxThresholdErr, r.Failures, r.Samples)
	}
	return r, nil
}
