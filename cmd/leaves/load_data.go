package main

import (
	"fmt"
	"os"

	"github.com/linkerlin/leaves/data"
)

// loadMatrix 按 NA 策略加载训练/评估数据（WP-20）。
// naPolicy: error（默认）| skip-row；非法值返回 usage 错误。
func loadMatrix(path, naPolicy string) (data.Matrix, error) {
	p, err := data.NormalizeNAPolicy(naPolicy)
	if err != nil {
		return nil, errUsage(fmt.Sprintf("--na-policy: %v", err))
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
