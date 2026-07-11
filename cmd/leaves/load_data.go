package main

import (
	"fmt"
	"os"

	"github.com/linkerlin/leaves/data"
)

// loadMatrix 按 NA 策略加载训练/评估数据（WP-20）。
// naPolicy: error（默认）| skip-row；非法值返回 usage 错误。
func loadMatrix(path, naPolicy string) (data.Matrix, error) {
	return loadMatrixOpts(path, naPolicy, 0)
}

// loadMatrixOpts numTarget≥2 时按多目标 CSV（末 N 列为标签）加载。
func loadMatrixOpts(path, naPolicy string, numTarget int) (data.Matrix, error) {
	p, err := data.NormalizeNAPolicy(naPolicy)
	if err != nil {
		return nil, errUsage(fmt.Sprintf("--na-policy: %v", err))
	}
	if numTarget >= 2 {
		opts := data.DefaultFileLoadOptions()
		opts.Format = data.FormatCSV
		opts.CSV.NAPolicy = p
		opts.CSV.NumTrailingTargets = numTarget
		opts.CSV.HasHeader = true
		if sniff, serr := data.SniffFileFormat(path); serr == nil {
			if sniff.Format == data.FormatLIBSVM || sniff.Format == data.FormatRanking {
				return nil, errAgent("data_load", "multi-target 仅支持数值 CSV/TSV（末 N 列标签）",
					"用 CSV：f0,f1,...,y0,y1 并设 --num-target N", false)
			}
			opts.CSV.HasHeader = sniff.CSV.HasHeader
			if sniff.CSV.Delim != 0 {
				opts.CSV.Delim = sniff.CSV.Delim
			}
			// TSV 嗅探常标 TSVLabelLast，多目标仍走 CSV 解析器
		}
		opts.CSV.NumTrailingTargets = numTarget
		opts.CSV.HasLabelColumn = false
		dm, err := data.FromFile(path, opts)
		if err != nil {
			return nil, errAgentWrap("data_load", fmt.Sprintf("load multi-target %s: %v", path, err),
				"CSV 末 --num-target 列为标签，其余为特征；须全数值", false, err)
		}
		if d, ok := dm.(*data.Dense); ok && d.SkippedRows > 0 {
			fmt.Fprintf(os.Stderr, "leaves: na-policy=skip-row 跳过 %d 行（缺失单元格）\n", d.SkippedRows)
		}
		return dm, nil
	}
	dm, err := data.FromFileAutoNA(path, p)
	if err != nil {
		return nil, errAgentWrap("data_load", fmt.Sprintf("load %s: %v", path, err),
			"数值 CSV/LIBSVM/ranking TSV；缺失用 --na-policy skip-row 丢行；非数值须预编码", false, err)
	}
	if d, ok := dm.(*data.Dense); ok && d.SkippedRows > 0 {
		fmt.Fprintf(os.Stderr, "leaves: na-policy=skip-row 跳过 %d 行（缺失单元格）\n", d.SkippedRows)
	}
	return dm, nil
}
