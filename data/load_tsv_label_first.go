package data

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"
)

// LoadDenseTSVLabelFirst 加载 TSV：首列为 label，其余为特征。
func LoadDenseTSVLabelFirst(path string) (*Dense, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	var rows [][]float64
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		parts := strings.Split(line, "\t")
		if len(parts) < 2 {
			return nil, fmt.Errorf("data: tsv row needs >=2 cols")
		}
		row := make([]float64, len(parts))
		for i, p := range parts {
			v, err := strconv.ParseFloat(strings.TrimSpace(p), 64)
			if err != nil {
				return nil, fmt.Errorf("data: parse %q: %w", p, err)
			}
			row[i] = v
		}
		rows = append(rows, row)
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	if len(rows) == 0 {
		return nil, fmt.Errorf("data: empty tsv %s", path)
	}
	cols := len(rows[0])
	n := len(rows)
	featCols := cols - 1
	vals := make([]float64, n*featCols)
	labels := make([]float64, n)
	for i, r := range rows {
		labels[i] = r[0]
		copy(vals[i*featCols:(i+1)*featCols], r[1:])
	}
	return NewDense(vals, n, featCols, labels, nil)
}
