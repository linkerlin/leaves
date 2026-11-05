// Package data 提供训练期数据容器。
package data

import (
	"fmt"
	"strconv"
	"strings"
)

// Matrix 训练数据矩阵（Dense MVP）。
type Matrix interface {
	NumRow() int
	NumCol() int
	Row(row int, buf []float64) error
	Labels() []float64
	Weights() []float64
}

// FeatureNamedMatrix 可选接口：提供特征列名。
type FeatureNamedMatrix interface {
	Matrix
	FeatureNames() []string
}

// Dense 行优先 Dense 矩阵。
type Dense struct {
	Data   []float64
	Rows   int
	Cols   int
	Y      []float64
	W      []float64
	FT     []FeatureType
	FNames []string
	// SkippedRows：CSV 在 NAPolicySkipRow 下因缺失跳过的行数（仅信息字段）。
	SkippedRows int
	// NaNCells：CSV 在 NAPolicyNaN 下转为 math.NaN() 的特征单元格数（WP-31）。
	NaNCells int
	// 多目标回归（LIB-21）：NTargets>1 时 YMulti 为行优先 [n*k]，Y 为第 0 目标副本。
	NTargets int
	YMulti   []float64
	// WP-22：列控制与缺失标签门禁元数据。
	MissingLabel bool
	HeaderCols   []string
	LabelName    string
	DroppedCols  []string
	IDName       string
	IDValues     []string
	WeightName   string
	// WP-31：字符串类别列自动编码字典（特征列名或特征序号 -> 类别字符串 -> 编码值）。
	CategoryMaps map[string]map[string]float64
}

// NewDense 创建 Dense 矩阵；vals 行优先 [rows*cols]。
func NewDense(vals []float64, rows, cols int, labels []float64, weights []float64) (*Dense, error) {
	if rows <= 0 || cols <= 0 {
		return nil, fmt.Errorf("data: invalid shape %dx%d", rows, cols)
	}
	if len(vals) != rows*cols {
		return nil, fmt.Errorf("data: vals len %d != %d", len(vals), rows*cols)
	}
	if len(labels) != rows {
		return nil, fmt.Errorf("data: labels len %d != rows %d", len(labels), rows)
	}
	if weights != nil && len(weights) != rows {
		return nil, fmt.Errorf("data: weights len %d != rows %d", len(weights), rows)
	}
	return &Dense{Data: vals, Rows: rows, Cols: cols, Y: labels, W: weights}, nil
}

func (d *Dense) NumRow() int { return d.Rows }
func (d *Dense) NumCol() int { return d.Cols }

func (d *Dense) Row(row int, buf []float64) error {
	if row < 0 || row >= d.Rows {
		return fmt.Errorf("data: row %d out of range", row)
	}
	if cap(buf) < d.Cols {
		buf = make([]float64, d.Cols)
	} else {
		buf = buf[:d.Cols]
	}
	off := row * d.Cols
	copy(buf, d.Data[off:off+d.Cols])
	return nil
}

// RowSlice 返回行视图（只读）。
func (d *Dense) RowSlice(row int) []float64 {
	off := row * d.Cols
	return d.Data[off : off+d.Cols]
}

func (d *Dense) Labels() []float64  { return d.Y }
func (d *Dense) Weights() []float64 { return d.W }

func (d *Dense) FeatureTypes() []FeatureType { return d.FT }
func (d *Dense) FeatureNames() []string      { return d.FNames }

// WeightAt 返回样本权重，缺省 1。
func WeightAt(dm Matrix, i int) float64 {
	w := dm.Weights()
	if w == nil || i >= len(w) {
		return 1.0
	}
	return w[i]
}

// HasSemanticFeatureNames 判断特征名列表是否包含真实表头列名（而非全部为空或默认占位符 f0/f1/Column_0）。
func HasSemanticFeatureNames(names []string) bool {
	if len(names) == 0 {
		return false
	}
	for i, raw := range names {
		s := strings.TrimSpace(raw)
		if s == "" {
			continue
		}
		lower := strings.ToLower(s)
		if lower == fmt.Sprintf("f%d", i) || lower == fmt.Sprintf("column_%d", i) || lower == fmt.Sprintf("feat_%d", i) {
			continue
		}
		if strings.HasPrefix(lower, "f") {
			if _, err := strconv.Atoi(lower[1:]); err == nil {
				continue
			}
		}
		return true
	}
	return false
}

// FeatureMismatchError 表示数据特征 Schema 与模型期望特征 Schema 不匹配（WP-40）。
type FeatureMismatchError struct {
	Expected []string
	Actual   []string
	Missing  []string
	Extra    []string
}

func (e *FeatureMismatchError) Error() string {
	if len(e.Missing) > 0 {
		return fmt.Sprintf("data: feature schema mismatch: missing features %v (expected %v, got %v)", e.Missing, e.Expected, e.Actual)
	}
	return fmt.Sprintf("data: feature count mismatch: expected %d features %v, got %d features %v", len(e.Expected), e.Expected, len(e.Actual), e.Actual)
}

// AlignToFeatures 按期望的 expected 特征名对齐、投影或重排当前 Dense 矩阵（WP-40）。
// 返回值：对齐后的 *Dense、是否发生了列重排/裁剪（reordered）、以及不匹配错误。
func (d *Dense) AlignToFeatures(expected []string) (*Dense, bool, error) {
	if d == nil || len(expected) == 0 {
		return d, false, nil
	}
	expSemantic := HasSemanticFeatureNames(expected)
	actSemantic := HasSemanticFeatureNames(d.FNames)

	if !expSemantic || !actSemantic {
		if d.Cols != len(expected) {
			return nil, false, &FeatureMismatchError{
				Expected: expected,
				Actual:   d.FNames,
			}
		}
		if expSemantic && len(d.FNames) == 0 {
			cp := *d
			cp.FNames = append([]string(nil), expected...)
			return &cp, false, nil
		}
		return d, false, nil
	}

	exactIdx := make(map[string]int, len(d.FNames))
	foldIdx := make(map[string]int, len(d.FNames))
	for i, name := range d.FNames {
		trimmed := strings.TrimSpace(name)
		if _, ok := exactIdx[trimmed]; !ok {
			exactIdx[trimmed] = i
		}
		lower := strings.ToLower(trimmed)
		if _, ok := foldIdx[lower]; !ok {
			foldIdx[lower] = i
		}
	}

	colOrder := make([]int, len(expected))
	usedActual := make(map[int]bool, len(expected))
	var missing []string
	identity := d.Cols == len(expected)
	for j, want := range expected {
		w := strings.TrimSpace(want)
		idx, ok := exactIdx[w]
		if !ok {
			idx, ok = foldIdx[strings.ToLower(w)]
		}
		if !ok {
			missing = append(missing, want)
			continue
		}
		colOrder[j] = idx
		usedActual[idx] = true
		if idx != j {
			identity = false
		}
	}

	if len(missing) > 0 {
		var extra []string
		for i, name := range d.FNames {
			if !usedActual[i] {
				extra = append(extra, name)
			}
		}
		return nil, false, &FeatureMismatchError{
			Expected: expected,
			Actual:   d.FNames,
			Missing:  missing,
			Extra:    extra,
		}
	}

	if identity {
		return d, false, nil
	}

	newCols := len(expected)
	newData := make([]float64, d.Rows*newCols)
	for r := 0; r < d.Rows; r++ {
		srcRow := d.Data[r*d.Cols : (r+1)*d.Cols]
		dstRow := newData[r*newCols : (r+1)*newCols]
		for j, srcCol := range colOrder {
			dstRow[j] = srcRow[srcCol]
		}
	}
	var newFT []FeatureType
	if len(d.FT) == d.Cols {
		newFT = make([]FeatureType, newCols)
		for j, srcCol := range colOrder {
			newFT[j] = d.FT[srcCol]
		}
	}
	cp := *d
	cp.Data = newData
	cp.Cols = newCols
	cp.FNames = append([]string(nil), expected...)
	cp.FT = newFT
	return &cp, true, nil
}

// AlignMatrixToFeatures 对 Matrix（*Dense 或 *DenseWithGroups）执行特征 Schema 对齐（WP-40）。
func AlignMatrixToFeatures(dm Matrix, expected []string) (Matrix, bool, error) {
	if dm == nil || len(expected) == 0 {
		return dm, false, nil
	}
	switch m := dm.(type) {
	case *Dense:
		return m.AlignToFeatures(expected)
	case *DenseWithGroups:
		if m.Dense == nil {
			return dm, false, nil
		}
		aligned, reordered, err := m.Dense.AlignToFeatures(expected)
		if err != nil {
			return nil, false, err
		}
		if !reordered {
			return dm, false, nil
		}
		cp := *m
		cp.Dense = aligned
		return &cp, true, nil
	default:
		if dm.NumCol() != len(expected) {
			return nil, false, &FeatureMismatchError{Expected: expected}
		}
		return dm, false, nil
	}
}
