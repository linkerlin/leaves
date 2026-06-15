package data

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
)

// LIBSVMOptions libsvm/svmlight 加载选项。
type LIBSVMOptions struct {
	HasLabel bool // 首列为标签（libsvm 惯例）
	NumCol   int  // 特征维；0 = 从数据推断 max(index)+1
	Limit    int  // 最大行数；0 = 不限
}

// FromLIBSVM 从 libsvm 文件加载 CSR 矩阵。
func FromLIBSVM(path string, opts LIBSVMOptions) (*CSR, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return FromLIBSVMReader(bufio.NewReader(f), opts)
}

// FromLIBSVMReader 从 Reader 解析 libsvm。
func FromLIBSVMReader(r io.Reader, opts LIBSVMOptions) (*CSR, error) {
	reader := bufio.NewReader(r)
	var indptr []int
	var cols []int
	var vals []float64
	var labels []float64
	maxCol := -1
	rows := 0

	indptr = append(indptr, 0)
	oneBased := false
	oneBasedSet := false
	for {
		line, err := reader.ReadString('\n')
		if err != nil && err != io.EOF {
			return nil, err
		}
		line = strings.TrimSpace(line)
		if line == "" {
			if err == io.EOF {
				break
			}
			continue
		}
		tokens := strings.Fields(line)
		if len(tokens) < 1 {
			return nil, fmt.Errorf("data: libsvm empty line")
		}
		start := 0
		if opts.HasLabel {
			y, err := strconv.ParseFloat(tokens[0], 64)
			if err != nil {
				return nil, fmt.Errorf("data: libsvm label: %w", err)
			}
			labels = append(labels, y)
			start = 1
		}
		for i := start; i < len(tokens); i++ {
			pair := strings.SplitN(tokens[i], ":", 2)
			if len(pair) != 2 {
				return nil, fmt.Errorf("data: libsvm token %q", tokens[i])
			}
			col, err := strconv.Atoi(pair[0])
			if err != nil {
				return nil, fmt.Errorf("data: libsvm col %q: %w", pair[0], err)
			}
			if col < 0 {
				return nil, fmt.Errorf("data: libsvm col index must be >= 0, got %d", col)
			}
			if !oneBasedSet && i == start {
				oneBased = col >= 1
				oneBasedSet = true
			}
			col0 := col
			if oneBased {
				col0 = col - 1
			}
			if col0 < 0 {
				return nil, fmt.Errorf("data: libsvm invalid col index %d", col)
			}
			v, err := strconv.ParseFloat(pair[1], 64)
			if err != nil {
				return nil, fmt.Errorf("data: libsvm val %q: %w", pair[1], err)
			}
			cols = append(cols, col0)
			vals = append(vals, v)
			if col0 > maxCol {
				maxCol = col0
			}
		}
		indptr = append(indptr, len(vals))
		rows++
		if opts.Limit > 0 && rows >= opts.Limit {
			break
		}
		if err == io.EOF {
			break
		}
	}
	if rows == 0 {
		return nil, fmt.Errorf("data: libsvm empty")
	}
	numCol := opts.NumCol
	if numCol <= 0 {
		numCol = maxCol + 1
	}
	if numCol <= 0 {
		return nil, fmt.Errorf("data: libsvm cannot infer num_col")
	}
	if !opts.HasLabel {
		labels = make([]float64, rows)
	}
	return NewCSR(indptr, cols, vals, numCol, labels, nil)
}
