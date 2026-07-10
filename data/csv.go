package data

import (
	"encoding/csv"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
)

// NA 策略（WP-20）：仅处理缺失，不做插补。
const (
	// NAPolicyError 遇缺失单元格报错（默认）。
	NAPolicyError = "error"
	// NAPolicySkipRow 跳过含缺失的整行。
	NAPolicySkipRow = "skip-row"
)

// CSVOptions CSV 加载选项。
type CSVOptions struct {
	HasHeader      bool
	HasLabelColumn bool // 为 true 时从 LabelCol 读取标签
	LabelCol       int  // 标签列索引（需 HasLabelColumn）
	Delim          rune
	SkipCols       []int
	// NAPolicy：error（默认）| skip-row。空串等同 error。
	// 缺失 = 空单元格或 nan/na/null/none/n/a/?（大小写不敏感）。
	// 非数值且非缺失 token 始终报错（不做类别编码）。
	NAPolicy string
}

// IsMissingCell 判断单元格是否为缺失标记（空或常见 NA token）。
func IsMissingCell(s string) bool {
	t := strings.TrimSpace(s)
	if t == "" {
		return true
	}
	switch strings.ToLower(t) {
	case "nan", "na", "null", "none", "n/a", "?":
		return true
	default:
		return false
	}
}

// NormalizeNAPolicy 校验并规范化策略；非法值返回 error。
func NormalizeNAPolicy(p string) (string, error) {
	switch strings.ToLower(strings.TrimSpace(p)) {
	case "", NAPolicyError:
		return NAPolicyError, nil
	case NAPolicySkipRow, "skip", "skiprow", "skip_row":
		return NAPolicySkipRow, nil
	default:
		return "", fmt.Errorf("data: invalid na-policy %q (want error|skip-row)", p)
	}
}

// FromCSV 从 CSV 文件加载 Dense 矩阵（数值列）。
func FromCSV(path string, opts CSVOptions) (*Dense, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return FromCSVReader(f, opts)
}

// FromCSVReader 从 Reader 解析 CSV。
func FromCSVReader(r io.Reader, opts CSVOptions) (*Dense, error) {
	policy, err := NormalizeNAPolicy(opts.NAPolicy)
	if err != nil {
		return nil, err
	}
	delim := ','
	if opts.Delim != 0 {
		delim = opts.Delim
	}
	reader := csv.NewReader(r)
	reader.Comma = delim
	reader.FieldsPerRecord = -1
	reader.LazyQuotes = true

	skip := make(map[int]bool)
	for _, c := range opts.SkipCols {
		skip[c] = true
	}
	if opts.HasLabelColumn && opts.LabelCol >= 0 {
		skip[opts.LabelCol] = true
	}

	var rows [][]float64
	var labels []float64
	var fnames []string
	first := true
	lineNo := 0
	skipped := 0
	for {
		rec, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("data: csv read: %w", err)
		}
		lineNo++
		if first && opts.HasHeader {
			for i, s := range rec {
				if !skip[i] {
					fnames = append(fnames, strings.TrimSpace(s))
				}
			}
			first = false
			continue
		}
		first = false

		feats, label, hasLabel, miss, perr := parseCSVRecord(rec, skip, opts)
		if perr != nil {
			return nil, fmt.Errorf("data: line %d: %w", lineNo, perr)
		}
		if miss {
			if policy == NAPolicySkipRow {
				skipped++
				continue
			}
			return nil, fmt.Errorf("data: line %d: missing value (empty cell or NA token); use na-policy=skip-row to drop row", lineNo)
		}
		if len(feats) == 0 {
			continue
		}
		rows = append(rows, feats)
		if opts.HasLabelColumn && opts.LabelCol >= 0 {
			if !hasLabel {
				// 标签列越界等：与缺失同等处理
				if policy == NAPolicySkipRow {
					rows = rows[:len(rows)-1]
					skipped++
					continue
				}
				return nil, fmt.Errorf("data: line %d: missing label", lineNo)
			}
			labels = append(labels, label)
		}
	}
	if len(rows) == 0 {
		if skipped > 0 {
			return nil, fmt.Errorf("data: csv empty after skipping %d rows with missing values", skipped)
		}
		return nil, fmt.Errorf("data: csv empty")
	}
	cols := len(rows[0])
	vals := make([]float64, len(rows)*cols)
	for i, row := range rows {
		if len(row) != cols {
			return nil, fmt.Errorf("data: row %d cols %d != %d", i, len(row), cols)
		}
		copy(vals[i*cols:(i+1)*cols], row)
	}
	if opts.HasLabelColumn && opts.LabelCol >= 0 && len(labels) != len(rows) {
		return nil, fmt.Errorf("data: labels %d != rows %d", len(labels), len(rows))
	}
	if !opts.HasLabelColumn {
		labels = make([]float64, len(rows))
	}
	d, err := NewDense(vals, len(rows), cols, labels, nil)
	if err != nil {
		return nil, err
	}
	d.FNames = fnames
	d.SkippedRows = skipped
	return d, nil
}

// parseCSVRecord 解析一行；miss=true 表示有缺失单元格。
func parseCSVRecord(rec []string, skip map[int]bool, opts CSVOptions) (feats []float64, label float64, hasLabel, miss bool, err error) {
	for i, s := range rec {
		cell := strings.TrimSpace(s)
		if skip[i] {
			if opts.HasLabelColumn && i == opts.LabelCol {
				if IsMissingCell(cell) {
					return nil, 0, false, true, nil
				}
				v, e := strconv.ParseFloat(cell, 64)
				if e != nil {
					return nil, 0, false, false, fmt.Errorf("label col %d %q: %w", i, s, e)
				}
				label = v
				hasLabel = true
			}
			continue
		}
		if IsMissingCell(cell) {
			return nil, 0, false, true, nil
		}
		v, e := strconv.ParseFloat(cell, 64)
		if e != nil {
			return nil, 0, false, false, fmt.Errorf("col %d %q: non-numeric: %w", i, s, e)
		}
		feats = append(feats, v)
	}
	return feats, label, hasLabel, false, nil
}
