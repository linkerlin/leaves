package data

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"
)

// LoadRankingTSV 加载排序 TSV：qid label feat1 feat2 ...
// 行须按 qid 连续分组（与 XGBoost qid 语义一致）。
func LoadRankingTSV(path, sep string) (*DenseWithGroups, error) {
	if sep == "" {
		sep = "\t"
	}
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var (
		rows    [][]float64
		labels  []float64
		curQid  = -1
		groups  []int
		curSize int
		nFeat   int
	)
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		parts := strings.Split(line, sep)
		if len(parts) < 3 {
			return nil, fmt.Errorf("data: ranking row needs qid label + features")
		}
		qid, err := strconv.Atoi(strings.TrimSpace(parts[0]))
		if err != nil {
			return nil, fmt.Errorf("data: bad qid %q", parts[0])
		}
		y, err := strconv.ParseFloat(strings.TrimSpace(parts[1]), 64)
		if err != nil {
			return nil, fmt.Errorf("data: bad label %q", parts[1])
		}
		feat := make([]float64, len(parts)-2)
		for i := range feat {
			feat[i], err = strconv.ParseFloat(strings.TrimSpace(parts[i+2]), 64)
			if err != nil {
				return nil, fmt.Errorf("data: bad feature col %d", i+2)
			}
		}
		if nFeat == 0 {
			nFeat = len(feat)
		} else if len(feat) != nFeat {
			return nil, fmt.Errorf("data: feature count mismatch")
		}
		if curQid < 0 {
			curQid = qid
		}
		if qid != curQid {
			groups = append(groups, curSize)
			curSize = 0
			curQid = qid
		}
		rows = append(rows, feat)
		labels = append(labels, y)
		curSize++
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}
	if curSize > 0 {
		groups = append(groups, curSize)
	}
	if len(rows) == 0 {
		return nil, fmt.Errorf("data: empty ranking file")
	}
	vals := make([]float64, len(rows)*nFeat)
	for i, row := range rows {
		copy(vals[i*nFeat:(i+1)*nFeat], row)
	}
	dense, err := NewDense(vals, len(rows), nFeat, labels, nil)
	if err != nil {
		return nil, err
	}
	return NewDenseWithGroups(dense, groups)
}
