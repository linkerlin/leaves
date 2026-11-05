package data

import (
	"encoding/csv"
	"fmt"
	"io"
	"math"
	"os"
	"sort"
	"strconv"
	"strings"
)

// NA 策略（WP-20 / WP-31）：
const (
	// NAPolicyError 遇缺失单元格报错（默认）。
	NAPolicyError = "error"
	// NAPolicySkipRow 跳过含缺失的整行。
	NAPolicySkipRow = "skip-row"
	// NAPolicyNaN 将特征列的缺失单元格转为 math.NaN() 交由 GBDT 树原生缺失方向处理（WP-31）。
	NAPolicyNaN = "nan"
)

// CSVOptions CSV 加载选项。
type CSVOptions struct {
	HasHeader      bool
	HasLabelColumn bool // 为 true 时从 LabelCol 读取标签
	LabelCol       int  // 标签列索引（需 HasLabelColumn）
	// NumTrailingTargets：≥2 时末 N 列为多目标标签（LIB-21；优先于单 LabelCol）。
	NumTrailingTargets int
	Delim              rune
	SkipCols           []int
	// NAPolicy：error（默认）| skip-row | nan。空串等同 error。
	// 缺失 = 空单元格或 nan/na/null/none/n/a/?（大小写不敏感）。
	NAPolicy string
	// LabelColName：按列名或 0-based 列序号指定标签列（WP-22；优先于 LabelCol）。
	LabelColName string
	// DropColNames：按列名或 0-based 列序号指定需丢弃的列（WP-22；如 ID、时间戳、文本列）。
	DropColNames []string
	// IDColName：按列名或 0-based 列序号指定样本 ID 列（WP-27；跳过特征并存入 Dense.IDValues）。
	IDColName string
	// WeightColName：按列名或 0-based 列序号指定样本权重列（WP-41；跳过特征并存入 Dense.W）。
	WeightColName string
	// AutoEncodeCats：当为 true 时，将特征列中的非数值字符串自动按字典序编码为 0, 1, 2...（WP-31）。
	AutoEncodeCats bool
	// CategoryMaps：预置的类别特征编码表（特征名或 f0/f1 -> token -> 编码值；用于验证/测试集与训练集对齐，未知类别映射为 NaN）。
	CategoryMaps map[string]map[string]float64
	// ExpectedFeatures：模型期望的特征列名列表（WP-40；有表头时自动跳过不在其中的多余列并按此顺序对齐，缺失时报错）。
	ExpectedFeatures []string
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
	case NAPolicyNaN, "keep-nan", "keep_nan":
		return NAPolicyNaN, nil
	default:
		return "", fmt.Errorf("data: invalid na-policy %q (want error|skip-row|nan)", p)
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

	nTrail := opts.NumTrailingTargets
	if nTrail >= 2 {
		return fromCSVMultiTarget(reader, opts, policy, nTrail)
	}

	skip := make(map[int]bool)
	for _, c := range opts.SkipCols {
		skip[c] = true
	}

	var rows [][]float64
	var labels []float64
	var weights []float64
	var idValues []string
	var fnames []string
	var headerCols []string
	var labelName string
	var droppedCols []string
	var idColName string
	var weightColName string
	idColIdx := -1
	weightColIdx := -1
	resolvedOpts := opts
	colsResolved := false
	first := true
	lineNo := 0
	skipped := 0
	nanCells := 0
	catMaps := make(map[string]map[string]float64)
	catColIdx := make(map[string]int)
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
			headerCols = make([]string, len(rec))
			for i, s := range rec {
				headerCols[i] = strings.TrimSpace(s)
			}
			var rerr error
			resolvedOpts, labelName, droppedCols, idColIdx, idColName, weightColIdx, weightColName, rerr = resolveCSVColumnOptions(opts, headerCols, len(rec), skip)
			if rerr != nil {
				return nil, rerr
			}
			colsResolved = true
			for i, s := range headerCols {
				if !skip[i] {
					fnames = append(fnames, s)
				}
			}
			first = false
			continue
		}
		first = false
		if !colsResolved {
			var rerr error
			resolvedOpts, labelName, droppedCols, idColIdx, idColName, weightColIdx, weightColName, rerr = resolveCSVColumnOptions(opts, nil, len(rec), skip)
			if rerr != nil {
				return nil, rerr
			}
			colsResolved = true
		}

		feats, label, hasLabel, idVal, weightVal, rowNaNs, miss, perr := parseCSVRecordWithCats(rec, skip, resolvedOpts, policy, idColIdx, weightColIdx, fnames, catMaps, catColIdx)
		if perr != nil {
			return nil, fmt.Errorf("data: line %d: %w", lineNo, perr)
		}
		if miss {
			if policy == NAPolicySkipRow {
				skipped++
				continue
			}
			return nil, fmt.Errorf("data: line %d: missing value (empty cell or NA token); use na-policy=skip-row or na-policy=nan", lineNo)
		}
		if len(feats) == 0 {
			continue
		}
		rows = append(rows, feats)
		nanCells += rowNaNs
		if resolvedOpts.HasLabelColumn && resolvedOpts.LabelCol >= 0 {
			if !hasLabel {
				// 标签列越界等：与缺失同等处理
				if policy == NAPolicySkipRow {
					rows = rows[:len(rows)-1]
					nanCells -= rowNaNs
					skipped++
					continue
				}
				return nil, fmt.Errorf("data: line %d: missing label", lineNo)
			}
			labels = append(labels, label)
		}
		if idColIdx >= 0 {
			idValues = append(idValues, idVal)
		}
		if weightColIdx >= 0 {
			weights = append(weights, weightVal)
		}
	}
	if len(rows) == 0 {
		if skipped > 0 {
			return nil, fmt.Errorf("data: csv empty after skipping %d rows with missing values", skipped)
		}
		return nil, fmt.Errorf("data: csv empty")
	}
	// WP-31：对新发现的字符串类别列按字典序重映射为确定性 0, 1, 2...
	for featKey, m := range catMaps {
		featIdx, ok := catColIdx[featKey]
		if !ok || len(m) == 0 {
			continue
		}
		tokens := make([]string, 0, len(m))
		for tok := range m {
			tokens = append(tokens, tok)
		}
		sort.Strings(tokens)
		remap := make(map[float64]float64, len(tokens))
		sortedMap := make(map[string]float64, len(tokens))
		for newIdx, tok := range tokens {
			oldVal := m[tok]
			remap[oldVal] = float64(newIdx)
			sortedMap[tok] = float64(newIdx)
		}
		for r := range rows {
			if featIdx < len(rows[r]) && !math.IsNaN(rows[r][featIdx]) {
				if nv, exists := remap[rows[r][featIdx]]; exists {
					rows[r][featIdx] = nv
				}
			}
		}
		catMaps[featKey] = sortedMap
	}
	// 合并预置的 CategoryMaps（若有）
	for k, v := range opts.CategoryMaps {
		if _, exists := catMaps[k]; !exists {
			catMaps[k] = v
		}
	}

	cols := len(rows[0])
	vals := make([]float64, len(rows)*cols)
	for i, row := range rows {
		if len(row) != cols {
			return nil, fmt.Errorf("data: row %d cols %d != %d", i, len(row), cols)
		}
		copy(vals[i*cols:(i+1)*cols], row)
	}
	if resolvedOpts.HasLabelColumn && resolvedOpts.LabelCol >= 0 && len(labels) != len(rows) {
		return nil, fmt.Errorf("data: labels %d != rows %d", len(labels), len(rows))
	}
	if !resolvedOpts.HasLabelColumn {
		labels = make([]float64, len(rows))
	}
	d, err := NewDense(vals, len(rows), cols, labels, weights)
	if err != nil {
		return nil, err
	}
	d.FNames = fnames
	d.SkippedRows = skipped
	d.NaNCells = nanCells
	d.MissingLabel = !resolvedOpts.HasLabelColumn
	d.HeaderCols = headerCols
	d.LabelName = labelName
	d.DroppedCols = droppedCols
	d.IDName = idColName
	d.IDValues = idValues
	d.WeightName = weightColName
	if len(catMaps) > 0 {
		d.CategoryMaps = catMaps
	}
	if len(opts.ExpectedFeatures) > 0 {
		aligned, _, aerr := d.AlignToFeatures(opts.ExpectedFeatures)
		if aerr != nil {
			return nil, aerr
		}
		d = aligned
	}
	return d, nil
}

// resolveColSpec 将列名或 0-based 整数列号解析为列索引与显示名。
func resolveColSpec(spec string, header []string, numCols int) (int, string, error) {
	s := strings.TrimSpace(spec)
	if s == "" {
		return -1, "", fmt.Errorf("data: empty column specification")
	}
	for i, h := range header {
		if h == s {
			return i, h, nil
		}
	}
	for i, h := range header {
		if strings.EqualFold(h, s) {
			return i, h, nil
		}
	}
	if idx, err := strconv.Atoi(s); err == nil {
		if idx < 0 || idx >= numCols {
			return -1, "", fmt.Errorf("data: column index %d out of range [0,%d)", idx, numCols)
		}
		name := s
		if idx < len(header) && header[idx] != "" {
			name = header[idx]
		}
		return idx, name, nil
	}
	if len(header) > 0 {
		return -1, "", fmt.Errorf("data: column %q not found in header %v", s, header)
	}
	return -1, "", fmt.Errorf("data: column %q cannot be resolved on headerless CSV (use 0-based index)", s)
}

func resolveCSVColumnOptions(opts CSVOptions, header []string, numCols int, skip map[int]bool) (CSVOptions, string, []string, int, string, int, string, error) {
	resolved := opts
	var labelName string
	var droppedCols []string
	idColIdx := -1
	var idColName string
	weightColIdx := -1
	var weightColName string

	if strings.TrimSpace(opts.LabelColName) != "" {
		idx, name, err := resolveColSpec(opts.LabelColName, header, numCols)
		if err != nil {
			return resolved, "", nil, -1, "", -1, "", err
		}
		resolved.HasLabelColumn = true
		resolved.LabelCol = idx
		labelName = name
	} else if resolved.HasLabelColumn && resolved.LabelCol >= 0 && resolved.LabelCol < len(header) {
		labelName = header[resolved.LabelCol]
	}

	if resolved.HasLabelColumn && resolved.LabelCol >= 0 {
		skip[resolved.LabelCol] = true
	}

	for _, raw := range opts.DropColNames {
		for _, part := range strings.Split(raw, ",") {
			part = strings.TrimSpace(part)
			if part == "" {
				continue
			}
			idx, name, err := resolveColSpec(part, header, numCols)
			if err != nil {
				return resolved, "", nil, -1, "", -1, "", err
			}
			skip[idx] = true
			droppedCols = append(droppedCols, name)
		}
	}

	if strings.TrimSpace(opts.IDColName) != "" {
		idx, name, err := resolveColSpec(opts.IDColName, header, numCols)
		if err != nil {
			return resolved, "", nil, -1, "", -1, "", err
		}
		idColIdx = idx
		idColName = name
		skip[idx] = true
	}

	if strings.TrimSpace(opts.WeightColName) != "" {
		idx, name, err := resolveColSpec(opts.WeightColName, header, numCols)
		if err != nil {
			return resolved, "", nil, -1, "", -1, "", err
		}
		weightColIdx = idx
		weightColName = name
		skip[idx] = true
	}

	// WP-40：若指定了模型语义特征名 ExpectedFeatures 且当前 CSV 有表头：
	// 1) 若启发式嗅探的默认 LabelCol（非用户显式 LabelColName）恰好是模型需要的特征列，则取消其 LabelCol 身份；
	// 2) 若表头已包含全部 ExpectedFeatures，则将不在 ExpectedFeatures 中的其余非特殊列自动记入 skip（免去推理时重复传 --drop-cols）。
	if len(header) > 0 && HasSemanticFeatureNames(opts.ExpectedFeatures) {
		expSet := make(map[string]bool, len(opts.ExpectedFeatures))
		for _, ef := range opts.ExpectedFeatures {
			expSet[strings.ToLower(strings.TrimSpace(ef))] = true
		}
		if strings.TrimSpace(opts.LabelColName) == "" && resolved.HasLabelColumn && resolved.LabelCol >= 0 && resolved.LabelCol < len(header) {
			lh := strings.ToLower(strings.TrimSpace(header[resolved.LabelCol]))
			if expSet[lh] {
				delete(skip, resolved.LabelCol)
				resolved.HasLabelColumn = false
				resolved.LabelCol = -1
				labelName = ""
			}
		}
		// 检查是否所有 ExpectedFeatures 都在未 skip 的 header 中存在
		availSet := make(map[string]bool, len(header))
		for i, h := range header {
			if !skip[i] {
				availSet[strings.ToLower(strings.TrimSpace(h))] = true
			}
		}
		allFound := true
		for _, ef := range opts.ExpectedFeatures {
			if !availSet[strings.ToLower(strings.TrimSpace(ef))] {
				allFound = false
				break
			}
		}
		if allFound {
			for i, h := range header {
				if skip[i] {
					continue
				}
				if !expSet[strings.ToLower(strings.TrimSpace(h))] {
					skip[i] = true
					droppedCols = append(droppedCols, h)
				}
			}
		}
	}

	return resolved, labelName, droppedCols, idColIdx, idColName, weightColIdx, weightColName, nil
}

// fromCSVMultiTarget 末 nTrail 列为多目标标签，其余为特征。
func fromCSVMultiTarget(reader *csv.Reader, opts CSVOptions, policy string, nTrail int) (*Dense, error) {
	var featRows [][]float64
	var targetRows [][]float64
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
			if len(rec) <= nTrail {
				return nil, fmt.Errorf("data: multi-target needs >%d columns, got %d", nTrail, len(rec))
			}
			for i := 0; i < len(rec)-nTrail; i++ {
				fnames = append(fnames, strings.TrimSpace(rec[i]))
			}
			first = false
			continue
		}
		first = false
		if len(rec) <= nTrail {
			return nil, fmt.Errorf("data: line %d: need >%d cols for multi-target, got %d", lineNo, nTrail, len(rec))
		}
		nFeat := len(rec) - nTrail
		feats := make([]float64, nFeat)
		tg := make([]float64, nTrail)
		miss := false
		for i := 0; i < len(rec); i++ {
			cell := strings.TrimSpace(rec[i])
			if IsMissingCell(cell) {
				miss = true
				break
			}
			v, e := strconv.ParseFloat(cell, 64)
			if e != nil {
				return nil, fmt.Errorf("data: line %d col %d %q: non-numeric: %w", lineNo, i, cell, e)
			}
			if i < nFeat {
				feats[i] = v
			} else {
				tg[i-nFeat] = v
			}
		}
		if miss {
			if policy == NAPolicySkipRow {
				skipped++
				continue
			}
			return nil, fmt.Errorf("data: line %d: missing value; use na-policy=skip-row", lineNo)
		}
		featRows = append(featRows, feats)
		targetRows = append(targetRows, tg)
	}
	if len(featRows) == 0 {
		return nil, fmt.Errorf("data: csv empty (multi-target)")
	}
	cols := len(featRows[0])
	vals := make([]float64, len(featRows)*cols)
	targets := make([]float64, len(featRows)*nTrail)
	for i := range featRows {
		if len(featRows[i]) != cols {
			return nil, fmt.Errorf("data: row %d cols mismatch", i)
		}
		copy(vals[i*cols:(i+1)*cols], featRows[i])
		copy(targets[i*nTrail:(i+1)*nTrail], targetRows[i])
	}
	d, err := NewMultiTargetDense(vals, len(featRows), cols, targets, nTrail, nil)
	if err != nil {
		return nil, err
	}
	d.FNames = fnames
	d.SkippedRows = skipped
	return d, nil
}

// parseCSVRecordWithCats 解析一行；支持 NAPolicyNaN、样本权重列与字符串类别特征编码（WP-31 / WP-41）。
func parseCSVRecordWithCats(
	rec []string,
	skip map[int]bool,
	opts CSVOptions,
	policy string,
	idColIdx int,
	weightColIdx int,
	fnames []string,
	catMaps map[string]map[string]float64,
	catColIdx map[string]int,
) (feats []float64, label float64, hasLabel bool, idVal string, weightVal float64, rowNaNs int, miss bool, err error) {
	weightVal = 1.0
	for i, s := range rec {
		cell := strings.TrimSpace(s)
		if skip[i] {
			if idColIdx >= 0 && i == idColIdx {
				if cell == "" {
					return nil, 0, false, "", 1.0, 0, true, nil
				}
				idVal = cell
			}
			if weightColIdx >= 0 && i == weightColIdx {
				if IsMissingCell(cell) {
					return nil, 0, false, "", 1.0, 0, true, nil
				}
				w, e := strconv.ParseFloat(cell, 64)
				if e != nil || math.IsNaN(w) || math.IsInf(w, 0) || w < 0 {
					return nil, 0, false, "", 1.0, 0, false, fmt.Errorf("weight col %d %q: must be non-negative finite number", i, s)
				}
				weightVal = w
			}
			if opts.HasLabelColumn && i == opts.LabelCol {
				if IsMissingCell(cell) {
					return nil, 0, false, "", 1.0, 0, true, nil
				}
				v, e := strconv.ParseFloat(cell, 64)
				if e != nil {
					return nil, 0, false, "", 1.0, 0, false, fmt.Errorf("label col %d %q: %w", i, s, e)
				}
				label = v
				hasLabel = true
			}
			continue
		}
		featIdx := len(feats)
		featKey := fmt.Sprintf("f%d", featIdx)
		if featIdx < len(fnames) && fnames[featIdx] != "" {
			featKey = fnames[featIdx]
		}
		if IsMissingCell(cell) {
			if policy == NAPolicyNaN {
				feats = append(feats, math.NaN())
				rowNaNs++
				continue
			}
			return nil, 0, false, "", 1.0, 0, true, nil
		}
		// 若该列在预置 CategoryMaps 中已声明为类别列，优先查表（防止数值型字符串 token 如 "001" 被当连续数或不一致处理）
		if len(opts.CategoryMaps) > 0 {
			cmap := opts.CategoryMaps[featKey]
			if cmap == nil {
				cmap = opts.CategoryMaps[fmt.Sprintf("f%d", featIdx)]
			}
			if cmap != nil {
				if v, ok := cmap[cell]; ok {
					feats = append(feats, v)
				} else {
					feats = append(feats, math.NaN())
					rowNaNs++
				}
				continue
			}
		}
		v, e := strconv.ParseFloat(cell, 64)
		if e != nil {
			if opts.AutoEncodeCats {
				m := catMaps[featKey]
				if m == nil {
					m = make(map[string]float64)
					catMaps[featKey] = m
					catColIdx[featKey] = featIdx
				}
				code, exists := m[cell]
				if !exists {
					code = float64(len(m))
					m[cell] = code
				}
				feats = append(feats, code)
				continue
			}
			return nil, 0, false, "", 1.0, 0, false, fmt.Errorf("col %d %q: non-numeric: %w", i, s, e)
		}
		feats = append(feats, v)
	}
	return feats, label, hasLabel, idVal, weightVal, rowNaNs, false, nil
}
