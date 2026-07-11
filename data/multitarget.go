package data

import "fmt"

// MultiTarget 多目标回归矩阵（LIB-21）。
// Targets 为行优先 [n_rows * NumTarget()]；与单标签 Labels() 并存时 Labels 为第 0 目标。
type MultiTarget interface {
	Matrix
	NumTarget() int
	// Targets 返回全部目标，长度 = NumRow()*NumTarget()。
	Targets() []float64
}

// AsMultiTarget 若矩阵支持多目标则返回接口。
func AsMultiTarget(dm Matrix) (MultiTarget, bool) {
	if dm == nil {
		return nil, false
	}
	if mt, ok := dm.(MultiTarget); ok && mt.NumTarget() > 1 {
		return mt, true
	}
	return nil, false
}

// NewMultiTargetDense 创建多目标 Dense。
// targets 长度须为 rows*nTargets，行优先 [y0_t0, y0_t1, ..., y1_t0, ...]。
func NewMultiTargetDense(vals []float64, rows, cols int, targets []float64, nTargets int, weights []float64) (*Dense, error) {
	if nTargets < 2 {
		return nil, fmt.Errorf("data: multi-target needs nTargets>=2, got %d", nTargets)
	}
	if rows <= 0 || cols <= 0 {
		return nil, fmt.Errorf("data: invalid shape %dx%d", rows, cols)
	}
	if len(vals) != rows*cols {
		return nil, fmt.Errorf("data: vals len %d != %d", len(vals), rows*cols)
	}
	if len(targets) != rows*nTargets {
		return nil, fmt.Errorf("data: targets len %d != rows*nTargets %d", len(targets), rows*nTargets)
	}
	if weights != nil && len(weights) != rows {
		return nil, fmt.Errorf("data: weights len %d != rows %d", len(weights), rows)
	}
	// Labels() 暴露第 0 目标，便于 sniff/兼容路径。
	y0 := make([]float64, rows)
	for i := 0; i < rows; i++ {
		y0[i] = targets[i*nTargets]
	}
	return &Dense{
		Data:     vals,
		Rows:     rows,
		Cols:     cols,
		Y:        y0,
		YMulti:   targets,
		NTargets: nTargets,
		W:        weights,
	}, nil
}

// NumTarget 目标维数；单目标 Dense 返回 1。
func (d *Dense) NumTarget() int {
	if d == nil || d.NTargets <= 1 {
		return 1
	}
	return d.NTargets
}

// Targets 多目标扁平标签；单目标时返回 Labels()。
func (d *Dense) Targets() []float64 {
	if d == nil {
		return nil
	}
	if d.NTargets > 1 && len(d.YMulti) == d.Rows*d.NTargets {
		return d.YMulti
	}
	return d.Y
}
