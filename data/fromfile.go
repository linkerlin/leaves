package data

import (
	"fmt"
	"path/filepath"
	"strings"
)

// FileLoadOptions 统一文件加载选项。
type FileLoadOptions struct {
	Format FileFormat
	CSV    CSVOptions
	LIBSVM LIBSVMOptions
}

// FromFile 按扩展名/内容嗅探加载训练矩阵（Dense、CSR 或 Grouped）。
func FromFile(path string, opts FileLoadOptions) (Matrix, error) {
	format := opts.Format
	if format == FormatAuto {
		sniff, err := SniffFileFormat(path)
		if err != nil {
			format = detectFileFormatByExt(path)
		} else {
			opts = mergeSniffedOpts(opts, sniff)
			format = opts.Format
		}
	}
	switch format {
	case FormatCSV:
		d, err := FromCSV(path, opts.CSV)
		if err != nil {
			return nil, err
		}
		return d, nil
	case FormatLIBSVM:
		return FromLIBSVM(path, opts.LIBSVM)
	case FormatRanking:
		sep := "\t"
		if strings.ToLower(filepath.Ext(path)) == ".csv" {
			sep = ","
		}
		return LoadRankingTSV(path, sep)
	case FormatTSVLabelLast:
		return LoadDenseTSV(path)
	default:
		return nil, fmt.Errorf("data: unsupported format for %q", path)
	}
}

// FromFileAuto 使用内容嗅探的默认选项加载。
func FromFileAuto(path string) (Matrix, error) {
	return FromFile(path, DefaultFileLoadOptions())
}

// DetectFileFormat 返回嗅探到的格式（失败时回退扩展名）。
func DetectFileFormat(path string) FileFormat {
	sniff, err := SniffFileFormat(path)
	if err != nil {
		return detectFileFormatByExt(path)
	}
	return sniff.Format
}
