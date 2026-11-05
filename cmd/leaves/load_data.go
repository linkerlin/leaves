package main

import (
	"errors"
	"fmt"
	"os"
	"strings"

	"github.com/linkerlin/leaves/v2/data"
)

// loadMatrix 按 NA 策略加载训练/评估数据（WP-20）。
// naPolicy: error（默认）| skip-row；非法值返回 usage 错误。
func loadMatrix(path, naPolicy string) (data.Matrix, error) {
	return loadMatrixWithCols(path, naPolicy, 0, "", "", "")
}

// loadMatrixOpts numTarget≥2 时按多目标 CSV（末 N 列为标签）加载。
func loadMatrixOpts(path, naPolicy string, numTarget int) (data.Matrix, error) {
	return loadMatrixWithCols(path, naPolicy, numTarget, "", "", "")
}

// loadMatrixWithCols 支持显式列控制（--label-col / --drop-cols / --id-col，WP-22/WP-27）。
func loadMatrixWithCols(path, naPolicy string, numTarget int, labelCol, dropCols, idCol string) (data.Matrix, error) {
	return loadMatrixWithColsAndCats(path, naPolicy, numTarget, labelCol, dropCols, idCol, false, nil)
}

// loadMatrixWithColsAndCats 支持显式列控制、--na-policy nan 与字符串类别编码（WP-22/WP-27/WP-31）。
func loadMatrixWithColsAndCats(
	path, naPolicy string,
	numTarget int,
	labelCol, dropCols, idCol string,
	autoEncode bool,
	catMaps map[string]map[string]float64,
) (data.Matrix, error) {
	return loadMatrixFull(path, naPolicy, numTarget, labelCol, dropCols, idCol, "", autoEncode, catMaps, nil)
}

// loadMatrixFull 支持显式列控制、样本权重列（--weight-col）、类别编码与模型特征 Schema 自动投影对齐（WP-40/WP-41）。
func loadMatrixFull(
	path, naPolicy string,
	numTarget int,
	labelCol, dropCols, idCol, weightCol string,
	autoEncode bool,
	catMaps map[string]map[string]float64,
	expectedFeatures []string,
) (data.Matrix, error) {
	p, err := data.NormalizeNAPolicy(naPolicy)
	if err != nil {
		return nil, errUsage(fmt.Sprintf("--na-policy: %v", err))
	}
	var dropList []string
	for _, part := range strings.Split(dropCols, ",") {
		part = strings.TrimSpace(part)
		if part != "" {
			dropList = append(dropList, part)
		}
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
	opts := data.DefaultFileLoadOptions()
	opts.CSV.NAPolicy = p
	opts.CSV.LabelColName = strings.TrimSpace(labelCol)
	opts.CSV.DropColNames = dropList
	opts.CSV.IDColName = strings.TrimSpace(idCol)
	opts.CSV.WeightColName = strings.TrimSpace(weightCol)
	opts.CSV.AutoEncodeCats = autoEncode
	opts.CSV.CategoryMaps = catMaps
	opts.CSV.ExpectedFeatures = expectedFeatures
	dm, err := data.FromFile(path, opts)
	if err != nil {
		var fmErr *data.FeatureMismatchError
		if errors.As(err, &fmErr) {
			return nil, errAgentWrap("feature_mismatch", fmt.Sprintf("load %s: %v", path, fmErr),
				"检查数据表头是否包含模型期望的全部特征列，或核对 --label-col / --drop-cols", false, err)
		}
		return nil, errAgentWrap("data_load", fmt.Sprintf("load %s: %v", path, err),
			"数值 CSV/LIBSVM/ranking TSV；缺失用 --na-policy skip-row|nan；非数值类别列用 --auto-encode 或 --drop-cols", false, err)
	}
	if len(expectedFeatures) > 0 {
		aligned, reordered, aerr := data.AlignMatrixToFeatures(dm, expectedFeatures)
		if aerr != nil {
			return nil, errAgentWrap("feature_mismatch", fmt.Sprintf("align %s: %v", path, aerr),
				"检查数据特征列与模型特征 Schema 是否一致", false, aerr)
		}
		if reordered {
			fmt.Fprintf(os.Stderr, "leaves: 已按模型特征 Schema 自动投影/重排 %d 列特征\n", len(expectedFeatures))
		}
		dm = aligned
	}
	if d, ok := dm.(*data.Dense); ok && d.SkippedRows > 0 {
		fmt.Fprintf(os.Stderr, "leaves: na-policy=skip-row 跳过 %d 行（缺失单元格）\n", d.SkippedRows)
	}
	return dm, nil
}

// requireLabelColumn 在 train/eval 入口校验：若 CSV 未识别出标签列且未通过 --label-col 指定，立即拦截报错（WP-22 / AN1）。
func requireLabelColumn(dm data.Matrix, path string, numTarget int) error {
	if numTarget >= 2 {
		return nil
	}
	if d, ok := dm.(*data.Dense); ok && d.MissingLabel {
		colsHint := ""
		if len(d.HeaderCols) > 0 {
			colsHint = fmt.Sprintf("（当前表头列：%s）", strings.Join(d.HeaderCols, ", "))
		}
		return errAgent(
			"missing_label",
			fmt.Sprintf("%s 未检测到标签列%s，禁止在全零伪标签上静默训练或评估", path, colsHint),
			fmt.Sprintf("请使用 --label-col <列名或0-based索引> 显式指定标签列%s", colsHint),
			false,
		)
	}
	return nil
}
