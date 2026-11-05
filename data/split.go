package data

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

// SplitOptions 数据集切分选项（WP-23）。
type SplitOptions struct {
	DataPath string
	OutDir   string
	Ratios   []float64 // 长度 2（train,val）或 3（train,val,test）
	Strategy string    // auto | random | stratified | group | time
	LabelCol string
	DropCols []string
	GroupCol string
	TimeCol  string
	Seed     int64
	NAPolicy string
}

// SplitReport 数据集切分产物报告（split.json）。
type SplitReport struct {
	SchemaVersion int                       `json:"schema_version"`
	Data          string                    `json:"data"`
	OutDir        string                    `json:"out_dir"`
	Format        string                    `json:"format"`
	Strategy      string                    `json:"strategy"`
	Ratios        []float64                 `json:"ratios"`
	Seed          int64                     `json:"seed"`
	TotalRows     int                       `json:"total_rows"`
	SkippedRows   int                       `json:"skipped_rows,omitempty"`
	MissingCells  int                       `json:"missing_cells,omitempty"`
	LeakageOK     bool                      `json:"leakage_ok"`
	TrainPath     string                    `json:"train_path"`
	ValPath       string                    `json:"val_path"`
	TestPath      string                    `json:"test_path,omitempty"`
	TrainRows     int                       `json:"train_rows"`
	ValRows       int                       `json:"val_rows"`
	TestRows      int                       `json:"test_rows,omitempty"`
	GroupCounts   map[string]int            `json:"group_counts,omitempty"`
	ClassCounts   map[string]map[string]int `json:"class_counts,omitempty"`
	LabelMeans    map[string]float64        `json:"label_means,omitempty"`
	MaxLabelDrift float64                   `json:"max_label_drift"`
}

// ParseSplitRatios 解析形如 "0.7,0.15,0.15" 或 "0.8,0.2" 的比例字符串。
func ParseSplitRatios(s string) ([]float64, error) {
	s = strings.TrimSpace(s)
	if s == "" {
		return []float64{0.7, 0.15, 0.15}, nil
	}
	parts := strings.Split(s, ",")
	if len(parts) != 2 && len(parts) != 3 {
		return nil, fmt.Errorf("data: split ratios must have 2 or 3 values (got %d in %q)", len(parts), s)
	}
	ratios := make([]float64, len(parts))
	sum := 0.0
	for i, p := range parts {
		v, err := strconv.ParseFloat(strings.TrimSpace(p), 64)
		if err != nil || v < 0 {
			return nil, fmt.Errorf("data: invalid split ratio %q", p)
		}
		ratios[i] = v
		sum += v
	}
	if sum <= 0 {
		return nil, fmt.Errorf("data: split ratios sum must be > 0")
	}
	for i := range ratios {
		ratios[i] /= sum
	}
	if len(ratios) == 3 && ratios[2] == 0 {
		ratios = ratios[:2]
	}
	return ratios, nil
}

// SplitDataset 执行零泄漏数据切分并落盘子集文件与 split.json。
func SplitDataset(opts SplitOptions) (*SplitReport, error) {
	if strings.TrimSpace(opts.DataPath) == "" {
		return nil, fmt.Errorf("data: split requires data path")
	}
	if strings.TrimSpace(opts.OutDir) == "" {
		return nil, fmt.Errorf("data: split requires out-dir")
	}
	policy, err := NormalizeNAPolicy(opts.NAPolicy)
	if err != nil {
		return nil, err
	}
	ratios := opts.Ratios
	if len(ratios) == 0 {
		ratios = []float64{0.7, 0.15, 0.15}
	}
	if len(ratios) != 2 && len(ratios) != 3 {
		return nil, fmt.Errorf("data: split ratios len must be 2 or 3")
	}
	if err := os.MkdirAll(opts.OutDir, 0o755); err != nil {
		return nil, err
	}

	sniff, err := SniffFileFormat(opts.DataPath)
	if err != nil {
		return nil, err
	}
	if sniff.Format == FormatLIBSVM {
		return splitLIBSVM(opts, ratios)
	}

	delim := ','
	hasHeader := false
	switch sniff.Format {
	case FormatRanking, FormatTSVLabelLast:
		delim = '\t'
		hasHeader = false
	default:
		if sniff.CSV.Delim != 0 {
			delim = sniff.CSV.Delim
		}
		hasHeader = sniff.CSV.HasHeader
	}

	f, err := os.Open(opts.DataPath)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	reader := csv.NewReader(f)
	reader.Comma = delim
	reader.FieldsPerRecord = -1
	reader.LazyQuotes = true

	var header []string
	var rows [][]string
	skipped := 0
	missingCells := 0
	first := true
	for {
		rec, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("data: split read csv: %w", err)
		}
		if first && hasHeader {
			header = make([]string, len(rec))
			for i, s := range rec {
				header[i] = strings.TrimSpace(s)
			}
			first = false
			continue
		}
		first = false
		if len(rec) == 0 {
			continue
		}
		rowMissing := 0
		for _, c := range rec {
			if IsMissingCell(c) {
				rowMissing++
			}
		}
		if rowMissing > 0 {
			switch policy {
			case NAPolicySkipRow:
				skipped++
				continue
			case NAPolicyNaN:
				missingCells += rowMissing
			default:
				return nil, fmt.Errorf("data: split encountered missing value; use na-policy=skip-row or na-policy=nan")
			}
		}
		cp := make([]string, len(rec))
		copy(cp, rec)
		rows = append(rows, cp)
	}
	if len(rows) == 0 {
		return nil, fmt.Errorf("data: split input has 0 valid rows")
	}
	numCols := len(rows[0])

	// 解析列索引
	labelIdx := -1
	if strings.TrimSpace(opts.LabelCol) != "" {
		idx, _, err := resolveColSpec(opts.LabelCol, header, numCols)
		if err != nil {
			return nil, err
		}
		labelIdx = idx
	} else if sniff.Format == FormatRanking && numCols >= 2 {
		labelIdx = 1
	} else if sniff.Format == FormatTSVLabelLast && numCols >= 1 {
		labelIdx = numCols - 1
	} else if sniff.CSV.HasLabelColumn && sniff.CSV.LabelCol >= 0 && sniff.CSV.LabelCol < numCols {
		labelIdx = sniff.CSV.LabelCol
	}

	groupIdx := -1
	if strings.TrimSpace(opts.GroupCol) != "" {
		idx, _, err := resolveColSpec(opts.GroupCol, header, numCols)
		if err != nil {
			return nil, err
		}
		groupIdx = idx
	} else if sniff.Format == FormatRanking {
		groupIdx = 0
	}

	timeIdx := -1
	if strings.TrimSpace(opts.TimeCol) != "" {
		idx, _, err := resolveColSpec(opts.TimeCol, header, numCols)
		if err != nil {
			return nil, err
		}
		timeIdx = idx
	}

	dropSet := make(map[int]bool)
	for _, raw := range opts.DropCols {
		for _, part := range strings.Split(raw, ",") {
			part = strings.TrimSpace(part)
			if part == "" {
				continue
			}
			idx, _, err := resolveColSpec(part, header, numCols)
			if err != nil {
				return nil, err
			}
			dropSet[idx] = true
		}
	}

	strategy := strings.ToLower(strings.TrimSpace(opts.Strategy))
	if strategy == "" || strategy == "auto" {
		switch {
		case timeIdx >= 0:
			strategy = "time"
		case groupIdx >= 0:
			strategy = "group"
		case labelIdx >= 0 && looksDiscreteClass(rows, labelIdx):
			strategy = "stratified"
		default:
			strategy = "random"
		}
	}

	k := len(ratios)
	subsets := make([][]int, k)
	rng := rand.New(rand.NewSource(opts.Seed))

	var groupCounts map[string]int
	var classCounts map[string]map[string]int

	switch strategy {
	case "random":
		idx := makeRange(len(rows))
		rng.Shuffle(len(idx), func(i, j int) { idx[i], idx[j] = idx[j], idx[i] })
		subsets = splitIndexSlice(idx, ratios)

	case "stratified":
		if labelIdx < 0 {
			return nil, fmt.Errorf("data: stratified split requires a detected or --label-col specified label column")
		}
		buckets := make(map[string][]int)
		var keys []string
		for i, r := range rows {
			lbl := strings.TrimSpace(r[labelIdx])
			if _, ok := buckets[lbl]; !ok {
				keys = append(keys, lbl)
			}
			buckets[lbl] = append(buckets[lbl], i)
		}
		sort.Strings(keys)
		classCounts = map[string]map[string]int{
			"train": {},
			"val":   {},
		}
		if k == 3 {
			classCounts["test"] = map[string]int{}
		}
		for _, key := range keys {
			b := buckets[key]
			rng.Shuffle(len(b), func(i, j int) { b[i], b[j] = b[j], b[i] })
			parts := splitIndexSlice(b, ratios)
			for si := 0; si < k; si++ {
				subsets[si] = append(subsets[si], parts[si]...)
			}
			classCounts["train"][key] = len(parts[0])
			classCounts["val"][key] = len(parts[1])
			if k == 3 {
				classCounts["test"][key] = len(parts[2])
			}
		}
		for si := 0; si < k; si++ {
			rng.Shuffle(len(subsets[si]), func(i, j int) {
				subsets[si][i], subsets[si][j] = subsets[si][j], subsets[si][i]
			})
		}

	case "group":
		if groupIdx < 0 {
			return nil, fmt.Errorf("data: group split requires --group-col or ranking TSV format")
		}
		groupBuckets := make(map[string][]int)
		var groupKeys []string
		for i, r := range rows {
			gk := strings.TrimSpace(r[groupIdx])
			if _, ok := groupBuckets[gk]; !ok {
				groupKeys = append(groupKeys, gk)
			}
			groupBuckets[gk] = append(groupBuckets[gk], i)
		}
		rng.Shuffle(len(groupKeys), func(i, j int) { groupKeys[i], groupKeys[j] = groupKeys[j], groupKeys[i] })
		groupParts := splitGroupKeys(groupKeys, groupBuckets, len(rows), ratios)
		groupCounts = map[string]int{
			"train": len(groupParts[0]),
			"val":   len(groupParts[1]),
		}
		if k == 3 {
			groupCounts["test"] = len(groupParts[2])
		}
		for si := 0; si < k; si++ {
			if sniff.Format == FormatRanking {
				// Ranking TSV 要求 qid 升序排列
				sort.SliceStable(groupParts[si], func(a, b int) bool {
					qa, errA := strconv.Atoi(groupParts[si][a])
					qb, errB := strconv.Atoi(groupParts[si][b])
					if errA == nil && errB == nil {
						return qa < qb
					}
					return groupParts[si][a] < groupParts[si][b]
				})
			}
			for _, gk := range groupParts[si] {
				subsets[si] = append(subsets[si], groupBuckets[gk]...)
			}
		}

	case "time":
		if timeIdx < 0 {
			return nil, fmt.Errorf("data: time split requires --time-col")
		}
		idx := makeRange(len(rows))
		sort.SliceStable(idx, func(i, j int) bool {
			sa := strings.TrimSpace(rows[idx[i]][timeIdx])
			sb := strings.TrimSpace(rows[idx[j]][timeIdx])
			fa, errA := strconv.ParseFloat(sa, 64)
			fb, errB := strconv.ParseFloat(sb, 64)
			if errA == nil && errB == nil {
				return fa < fb
			}
			return sa < sb
		})
		subsets = splitIndexSlice(idx, ratios)

	default:
		return nil, fmt.Errorf("data: unknown split strategy %q (want auto|random|stratified|group|time)", opts.Strategy)
	}

	ext := ".csv"
	formatName := "csv"
	if sniff.Format == FormatRanking {
		ext = ".tsv"
		formatName = "ranking_tsv"
	} else if sniff.Format == FormatTSVLabelLast || delim == '\t' {
		if strings.EqualFold(filepath.Ext(opts.DataPath), ".tsv") {
			ext = ".tsv"
		}
		formatName = "tsv"
	}

	filterRec := func(rec []string) []string {
		if len(dropSet) == 0 {
			return rec
		}
		out := make([]string, 0, len(rec)-len(dropSet))
		for i, c := range rec {
			if !dropSet[i] {
				out = append(out, c)
			}
		}
		return out
	}

	var outHeader []string
	if hasHeader {
		outHeader = filterRec(header)
	}

	trainPath := filepath.Join(opts.OutDir, "train"+ext)
	valPath := filepath.Join(opts.OutDir, "val"+ext)
	if err := writeCSVSubset(trainPath, delim, outHeader, rows, subsets[0], filterRec); err != nil {
		return nil, err
	}
	if err := writeCSVSubset(valPath, delim, outHeader, rows, subsets[1], filterRec); err != nil {
		return nil, err
	}
	var testPath string
	testRows := 0
	if k == 3 {
		testPath = filepath.Join(opts.OutDir, "test"+ext)
		if err := writeCSVSubset(testPath, delim, outHeader, rows, subsets[2], filterRec); err != nil {
			return nil, err
		}
		testRows = len(subsets[2])
	}

	labelMeans, maxDrift := computeSubsetLabelMeans(rows, subsets, labelIdx)

	rep := &SplitReport{
		SchemaVersion: 1,
		Data:          opts.DataPath,
		OutDir:        opts.OutDir,
		Format:        formatName,
		Strategy:      strategy,
		Ratios:        ratios,
		Seed:          opts.Seed,
		TotalRows:     len(rows),
		SkippedRows:   skipped,
		MissingCells:  missingCells,
		LeakageOK:     true,
		TrainPath:     trainPath,
		ValPath:       valPath,
		TestPath:      testPath,
		TrainRows:     len(subsets[0]),
		ValRows:       len(subsets[1]),
		TestRows:      testRows,
		GroupCounts:   groupCounts,
		ClassCounts:   classCounts,
		LabelMeans:    labelMeans,
		MaxLabelDrift: maxDrift,
	}
	repBytes, err := json.MarshalIndent(rep, "", "  ")
	if err != nil {
		return nil, err
	}
	if err := os.WriteFile(filepath.Join(opts.OutDir, "split.json"), append(repBytes, '\n'), 0o644); err != nil {
		return nil, err
	}
	return rep, nil
}

func computeSubsetLabelMeans(rows [][]string, subsets [][]int, labelIdx int) (map[string]float64, float64) {
	if labelIdx < 0 || len(subsets) < 2 {
		return nil, 0
	}
	names := []string{"train", "val", "test"}
	means := make(map[string]float64)
	minMean := math.Inf(1)
	maxMean := math.Inf(-1)
	for si, sub := range subsets {
		if si >= len(names) || len(sub) == 0 {
			continue
		}
		sum := 0.0
		cnt := 0
		for _, rIdx := range sub {
			if rIdx < 0 || rIdx >= len(rows) || labelIdx >= len(rows[rIdx]) {
				continue
			}
			v, err := strconv.ParseFloat(strings.TrimSpace(rows[rIdx][labelIdx]), 64)
			if err != nil || math.IsNaN(v) || math.IsInf(v, 0) {
				continue
			}
			sum += v
			cnt++
		}
		if cnt > 0 {
			m := math.Round((sum/float64(cnt))*10000) / 10000
			means[names[si]] = m
			if m < minMean {
				minMean = m
			}
			if m > maxMean {
				maxMean = m
			}
		}
	}
	if len(means) == 0 {
		return nil, 0
	}
	drift := 0.0
	if len(means) >= 2 && maxMean >= minMean {
		drift = math.Round((maxMean-minMean)*10000) / 10000
	}
	return means, drift
}

func looksDiscreteClass(rows [][]string, labelIdx int) bool {
	seen := make(map[string]bool)
	for _, r := range rows {
		if labelIdx >= len(r) {
			return false
		}
		s := strings.TrimSpace(r[labelIdx])
		v, err := strconv.ParseFloat(s, 64)
		if err != nil || v != math.Trunc(v) || v < 0 {
			return false
		}
		seen[s] = true
		if len(seen) > 20 {
			return false
		}
	}
	return len(seen) >= 2
}

// splitIndexSlice 将 idx 按 ratios 切分为 2 或 3 段，确保总长严格等于 len(idx)。
func splitIndexSlice(idx []int, ratios []float64) [][]int {
	n := len(idx)
	k := len(ratios)
	counts := make([]int, k)
	rem := n
	for i := 0; i < k-1; i++ {
		c := int(math.Round(float64(n) * ratios[i]))
		if c == 0 && ratios[i] > 0 && rem > (k-1-i) {
			c = 1
		}
		if c > rem-(k-1-i) && n >= k {
			c = rem - (k - 1 - i)
		}
		if c < 0 {
			c = 0
		}
		counts[i] = c
		rem -= c
	}
	counts[k-1] = rem

	out := make([][]int, k)
	off := 0
	for i := 0; i < k; i++ {
		out[i] = idx[off : off+counts[i]]
		off += counts[i]
	}
	return out
}

// splitGroupKeys 将组键切分为 k 个不交子集，保证任何组都不跨集。
func splitGroupKeys(groupKeys []string, buckets map[string][]int, totalRows int, ratios []float64) [][]string {
	k := len(ratios)
	out := make([][]string, k)
	if len(groupKeys) <= k {
		for i, gk := range groupKeys {
			out[i%k] = append(out[i%k], gk)
		}
		return out
	}
	targets := make([]int, k)
	for i := 0; i < k; i++ {
		targets[i] = int(math.Round(float64(totalRows) * ratios[i]))
	}
	cur := 0
	curRows := 0
	for i, gk := range groupKeys {
		remGroups := len(groupKeys) - i
		remSubsets := k - 1 - cur
		if cur < k-1 && remGroups <= remSubsets {
			cur++
			curRows = 0
		}
		out[cur] = append(out[cur], gk)
		curRows += len(buckets[gk])
		if cur < k-1 && curRows >= targets[cur] && (len(groupKeys)-i-1) >= (k-1-cur) {
			cur++
			curRows = 0
		}
	}
	return out
}

func writeCSVSubset(path string, delim rune, header []string, rows [][]string, indices []int, filter func([]string) []string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	w := csv.NewWriter(f)
	w.Comma = delim
	if len(header) > 0 {
		if err := w.Write(header); err != nil {
			return err
		}
	}
	for _, idx := range indices {
		if err := w.Write(filter(rows[idx])); err != nil {
			return err
		}
	}
	w.Flush()
	return w.Error()
}

func splitLIBSVM(opts SplitOptions, ratios []float64) (*SplitReport, error) {
	raw, err := os.ReadFile(opts.DataPath)
	if err != nil {
		return nil, err
	}
	var lines []string
	for _, line := range strings.Split(string(raw), "\n") {
		t := strings.TrimSpace(line)
		if t == "" || strings.HasPrefix(t, "#") {
			continue
		}
		lines = append(lines, t)
	}
	if len(lines) == 0 {
		return nil, fmt.Errorf("data: split libsvm empty")
	}
	rng := rand.New(rand.NewSource(opts.Seed))
	idx := makeRange(len(lines))
	rng.Shuffle(len(idx), func(i, j int) { idx[i], idx[j] = idx[j], idx[i] })
	parts := splitIndexSlice(idx, ratios)

	writeLines := func(p string, sub []int) error {
		var b strings.Builder
		for _, i := range sub {
			b.WriteString(lines[i])
			b.WriteByte('\n')
		}
		return os.WriteFile(p, []byte(b.String()), 0o644)
	}
	trainPath := filepath.Join(opts.OutDir, "train.libsvm")
	valPath := filepath.Join(opts.OutDir, "val.libsvm")
	if err := writeLines(trainPath, parts[0]); err != nil {
		return nil, err
	}
	if err := writeLines(valPath, parts[1]); err != nil {
		return nil, err
	}
	var testPath string
	testRows := 0
	if len(ratios) == 3 {
		testPath = filepath.Join(opts.OutDir, "test.libsvm")
		if err := writeLines(testPath, parts[2]); err != nil {
			return nil, err
		}
		testRows = len(parts[2])
	}
	rep := &SplitReport{
		SchemaVersion: 1,
		Data:          opts.DataPath,
		OutDir:        opts.OutDir,
		Format:        "libsvm",
		Strategy:      "random",
		Ratios:        ratios,
		Seed:          opts.Seed,
		TotalRows:     len(lines),
		LeakageOK:     true,
		TrainPath:     trainPath,
		ValPath:       valPath,
		TestPath:      testPath,
		TrainRows:     len(parts[0]),
		ValRows:       len(parts[1]),
		TestRows:      testRows,
	}
	repBytes, err := json.MarshalIndent(rep, "", "  ")
	if err != nil {
		return nil, err
	}
	if err := os.WriteFile(filepath.Join(opts.OutDir, "split.json"), append(repBytes, '\n'), 0o644); err != nil {
		return nil, err
	}
	return rep, nil
}
