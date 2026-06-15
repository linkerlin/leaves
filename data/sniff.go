package data

import (
	"bufio"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

const sniffMaxBytes = 64 * 1024

// FileFormat 训练数据文件格式。
type FileFormat int

const (
	FormatAuto FileFormat = iota
	FormatCSV
	FormatLIBSVM
	// FormatRanking 排序 TSV：qid label feat1 feat2 ...
	FormatRanking
	// FormatTSVLabelLast TSV/空格分隔，末列为 label。
	FormatTSVLabelLast
)

// SniffResult 内容嗅探结果。
type SniffResult struct {
	Format FileFormat
	CSV    CSVOptions
	LIBSVM LIBSVMOptions
}

// SniffFileFormat 读文件样本，自动推断训练数据格式与默认选项。
func SniffFileFormat(path string) (SniffResult, error) {
	sample, err := readFileSample(path, sniffMaxBytes)
	if err != nil {
		return SniffResult{}, err
	}
	ext := strings.ToLower(filepath.Ext(path))
	if looksLikeLIBSVM(sample) {
		return SniffResult{
			Format: FormatLIBSVM,
			LIBSVM: LIBSVMOptions{HasLabel: true},
		}, nil
	}
	if looksLikeRankingTSV(sample) {
		return SniffResult{Format: FormatRanking}, nil
	}
	csvOpts := sniffCSVOptions(sample, ext)
	if csvOpts.delim == '\t' && csvOpts.labelLast && !csvOpts.hasHeader {
		return SniffResult{Format: FormatTSVLabelLast}, nil
	}
	return SniffResult{
		Format: FormatCSV,
		CSV:    csvOpts.toCSVOptions(),
	}, nil
}

type csvSniff struct {
	delim      rune
	hasHeader  bool
	labelLast  bool
	labelCol   int
	hasLabel   bool
}

func (c csvSniff) toCSVOptions() CSVOptions {
	opts := CSVOptions{Delim: c.delim}
	if c.hasHeader {
		opts.HasHeader = true
	}
	if c.hasLabel {
		opts.HasLabelColumn = true
		opts.LabelCol = c.labelCol
	}
	return opts
}

func sniffCSVOptions(sample []byte, ext string) csvSniff {
	lines := nonEmptyLines(sample, 5)
	if len(lines) == 0 {
		out := csvSniff{delim: ','}
		if ext == ".tsv" {
			out.delim = '\t'
		}
		return out
	}
	delim := sniffDelimiter(lines[0], ext)
	fields := splitFields(lines[0], delim)
	out := csvSniff{delim: delim}

	if len(lines) > 0 && !lineAllNumeric(fields) {
		out.hasHeader = true
		out.hasLabel, out.labelCol = sniffLabelColumn(fields)
		return out
	}
	// 无表头：末列 label 启发式（末列整数/小范围且特征列更多）
	if len(lines) >= 2 && len(fields) >= 2 {
		if lastColLooksLikeLabel(lines, delim) {
			out.hasLabel = true
			out.labelCol = len(fields) - 1
			out.labelLast = true
		}
	}
	return out
}

func sniffDelimiter(line string, ext string) rune {
	if ext == ".tsv" {
		return '\t'
	}
	commas := strings.Count(line, ",")
	tabs := strings.Count(line, "\t")
	if tabs > commas {
		return '\t'
	}
	return ','
}

func splitFields(line string, delim rune) []string {
	if delim == '\t' {
		return strings.Split(line, "\t")
	}
	return strings.Split(line, ",")
}

func lineAllNumeric(fields []string) bool {
	if len(fields) == 0 {
		return false
	}
	for _, f := range fields {
		f = strings.TrimSpace(f)
		if f == "" {
			return false
		}
		if _, err := strconv.ParseFloat(f, 64); err != nil {
			return false
		}
	}
	return true
}

func sniffLabelColumn(header []string) (bool, int) {
	for i, h := range header {
		k := strings.ToLower(strings.TrimSpace(h))
		switch k {
		case "label", "target", "y", "class", "response":
			return true, i
		}
	}
	return false, -1
}

func lastColLooksLikeLabel(lines []string, delim rune) bool {
	if len(lines) < 2 {
		return false
	}
	f0 := splitFields(lines[0], delim)
	if len(f0) < 3 {
		return false
	}
	lastIdx := len(f0) - 1
	for _, line := range lines {
		fs := splitFields(line, delim)
		if len(fs) != len(f0) {
			return false
		}
		if _, err := strconv.ParseFloat(strings.TrimSpace(fs[lastIdx]), 64); err != nil {
			return false
		}
		for j := 0; j < lastIdx; j++ {
			if _, err := strconv.ParseFloat(strings.TrimSpace(fs[j]), 64); err != nil {
				return false
			}
		}
	}
	return true
}

func looksLikeLIBSVM(sample []byte) bool {
	lines := nonEmptyLines(sample, 8)
	if len(lines) == 0 {
		return false
	}
	hits := 0
	for _, line := range lines {
		if strings.HasPrefix(line, "#") {
			continue
		}
		if libsvmLinePattern(line) {
			hits++
		}
	}
	return hits >= 1 && hits*2 >= len(lines)
}

func libsvmLinePattern(line string) bool {
	tokens := strings.Fields(line)
	if len(tokens) < 2 {
		return false
	}
	start := 0
	if _, err := strconv.ParseFloat(tokens[0], 64); err == nil {
		if strings.Contains(tokens[1], ":") {
			start = 1
		} else if len(tokens) >= 3 && strings.Contains(tokens[2], ":") {
			start = 2
		}
	}
	pairs := 0
	for i := start; i < len(tokens); i++ {
		if strings.Contains(tokens[i], ":") {
			parts := strings.SplitN(tokens[i], ":", 2)
			if len(parts) == 2 {
				if _, err := strconv.Atoi(parts[0]); err == nil {
					if _, err := strconv.ParseFloat(parts[1], 64); err == nil {
						pairs++
					}
				}
			}
		}
	}
	return pairs >= 1
}

func looksLikeRankingTSV(sample []byte) bool {
	lines := nonEmptyLines(sample, 12)
	if len(lines) < 2 {
		return false
	}
	delim := sniffDelimiter(lines[0], ".tsv")
	hits := 0
	prevQid := -1
	seenQid := false
	for _, line := range lines {
		if strings.HasPrefix(line, "#") {
			continue
		}
		parts := splitFields(line, delim)
		if len(parts) < 3 {
			return false
		}
		qid, err := strconv.Atoi(strings.TrimSpace(parts[0]))
		if err != nil {
			return false
		}
		if _, err := strconv.ParseFloat(strings.TrimSpace(parts[1]), 64); err != nil {
			return false
		}
		for i := 2; i < len(parts); i++ {
			if _, err := strconv.ParseFloat(strings.TrimSpace(parts[i]), 64); err != nil {
				return false
			}
		}
		if seenQid && qid < prevQid {
			return false
		}
		prevQid = qid
		seenQid = true
		hits++
	}
	return hits >= 2
}

func nonEmptyLines(sample []byte, max int) []string {
	var out []string
	sc := bufio.NewScanner(strings.NewReader(string(sample)))
	for sc.Scan() && len(out) < max {
		line := strings.TrimSpace(sc.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		out = append(out, line)
	}
	return out
}

func readFileSample(path string, max int) ([]byte, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	buf := make([]byte, max)
	n, err := f.Read(buf)
	if err != nil && n == 0 {
		return nil, err
	}
	return buf[:n], nil
}

func detectFileFormatByExt(path string) FileFormat {
	ext := strings.ToLower(filepath.Ext(path))
	switch ext {
	case ".libsvm", ".svmlight", ".svm":
		return FormatLIBSVM
	case ".tsv":
		return FormatCSV
	case ".csv", ".txt":
		return FormatCSV
	default:
		return FormatCSV
	}
}

func mergeSniffedOpts(user FileLoadOptions, sniff SniffResult) FileLoadOptions {
	if user.Format != FormatAuto {
		return user
	}
	out := user
	out.Format = sniff.Format
	switch sniff.Format {
	case FormatLIBSVM:
		if !user.libsvmConfigured() {
			out.LIBSVM = sniff.LIBSVM
		}
	case FormatCSV:
		if !user.csvConfigured() {
			out.CSV = sniff.CSV
		} else if out.CSV.Delim == 0 && sniff.CSV.Delim != 0 {
			out.CSV.Delim = sniff.CSV.Delim
		}
	}
	return out
}

func (o FileLoadOptions) csvConfigured() bool {
	return o.CSV.HasHeader || o.CSV.HasLabelColumn || o.CSV.Delim != 0 || len(o.CSV.SkipCols) > 0
}

func (o FileLoadOptions) libsvmConfigured() bool {
	return o.LIBSVM.HasLabel || o.LIBSVM.NumCol > 0 || o.LIBSVM.Limit > 0
}

// DefaultFileLoadOptions 返回启用自动嗅探的默认选项。
func DefaultFileLoadOptions() FileLoadOptions {
	return FileLoadOptions{Format: FormatAuto}
}
