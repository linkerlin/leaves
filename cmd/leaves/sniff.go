package main

import (
	"flag"
	"fmt"

	"github.com/linkerlin/leaves/data"
)

func cmdSniff(args []string) error {
	fs := flag.NewFlagSet("sniff", flag.ContinueOnError)
	fs.Usage = func() { fmt.Fprintln(fs.Output(), "用法: leaves sniff --data PATH [--metrics PATH]") }
	dataPath := fs.String("data", "", "数据路径（必需）")
	metricsPath := fs.String("metrics", "", "输出 JSON（空=stdout）")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *dataPath == "" {
		return errUsage("--data 必需")
	}

	sniff, err := data.SniffFileFormat(*dataPath)
	if err != nil {
		return fmt.Errorf("sniff: %w", err)
	}
	dm, err := data.FromFileAuto(*dataPath)
	if err != nil {
		return fmt.Errorf("load: %w", err)
	}

	nRows, nCols := dm.NumRow(), dm.NumCol()
	ls := labelStats(dm.Labels())
	isRank := sniff.Format == data.FormatRanking
	nFeat := nCols
	hasLabel := len(dm.Labels()) == nRows
	if hasLabel && !isRank {
		nFeat = nCols - 1
	} else if isRank {
		nFeat = nCols - 2 // qid + label
	}

	suggObj, suggMetric := suggestObjective(ls, sniff.Format)
	doc := map[string]any{
		"file":                *dataPath,
		"format":              formatName(sniff.Format),
		"n_rows":              nRows,
		"n_cols":              nCols,
		"n_features":          nFeat,
		"has_label":           hasLabel || isRank,
		"label":               ls,
		"suggested_objective": suggObj,
		"suggested_metric":    suggMetric,
	}
	if d, ok := dm.(*data.Dense); ok && len(d.FNames) > 0 {
		doc["feature_names"] = d.FNames
	}
	return writeJSON(*metricsPath, doc)
}

func labelStats(labels []float64) map[string]any {
	if len(labels) == 0 {
		return map[string]any{"detected": false}
	}
	minV, maxV := labels[0], labels[0]
	uniq := map[float64]struct{}{}
	allInt := true
	for _, y := range labels {
		if y < minV {
			minV = y
		}
		if y > maxV {
			maxV = y
		}
		uniq[y] = struct{}{}
		if y != float64(int64(y)) {
			allInt = false
		}
	}
	kind := "regression"
	if allInt {
		switch {
		case len(uniq) == 2 && minV >= 0 && maxV <= 1:
			kind = "binary"
		case len(uniq) <= 20:
			kind = "classification"
		}
	}
	return map[string]any{
		"detected": true, "min": minV, "max": maxV,
		"n_unique": len(uniq), "kind": kind,
	}
}

func suggestObjective(ls map[string]any, format data.FileFormat) (string, string) {
	if format == data.FormatRanking {
		return "rank:ndcg", "ndcg@10"
	}
	kind, _ := ls["kind"].(string)
	switch kind {
	case "binary":
		return "binary:logistic", "logloss"
	case "classification":
		return "multi:softmax", "mlogloss"
	default:
		return "reg:squarederror", "rmse"
	}
}

func formatName(f data.FileFormat) string {
	switch f {
	case data.FormatCSV:
		return "csv"
	case data.FormatLIBSVM:
		return "libsvm"
	case data.FormatRanking:
		return "ranking"
	case data.FormatTSVLabelLast:
		return "tsv_label_last"
	default:
		return "auto"
	}
}
