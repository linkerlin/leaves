package data

import (
	"encoding/csv"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
)

// CSVOptions CSV 加载选项。
type CSVOptions struct {
	HasHeader      bool
	HasLabelColumn bool // 为 true 时从 LabelCol 读取标签
	LabelCol       int  // 标签列索引（需 HasLabelColumn）
	Delim          rune
	SkipCols       []int
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
	first := true
	for {
		rec, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("data: csv read: %w", err)
		}
		if first && opts.HasHeader {
			first = false
			continue
		}
		first = false
		var feats []float64
		for i, s := range rec {
			if skip[i] {
				if opts.HasLabelColumn && i == opts.LabelCol {
					v, err := strconv.ParseFloat(strings.TrimSpace(s), 64)
					if err != nil {
						return nil, fmt.Errorf("data: label col %d: %w", i, err)
					}
					labels = append(labels, v)
				}
				continue
			}
			v, err := strconv.ParseFloat(strings.TrimSpace(s), 64)
			if err != nil {
				return nil, fmt.Errorf("data: col %d %q: %w", i, s, err)
			}
			feats = append(feats, v)
		}
		if len(feats) == 0 {
			continue
		}
		rows = append(rows, feats)
	}
	if len(rows) == 0 {
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
	return NewDense(vals, len(rows), cols, labels, nil)
}
