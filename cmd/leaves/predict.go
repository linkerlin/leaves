package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"

	leavesio "github.com/linkerlin/leaves/v2/io"
)

func cmdPredict(args []string) error {
	fs := flag.NewFlagSet("predict", flag.ContinueOnError)
	fs.Usage = func() { fmt.Fprintln(fs.Output(), "用法: leaves predict --model PATH --data PATH --out PATH") }
	modelPath := fs.String("model", "", "模型路径（必需）")
	dataPath := fs.String("data", "", "数据路径（必需）")
	out := fs.String("out", "", "输出 JSONL 路径（必需）")
	objective := fs.String("objective", "", "目标函数；binary:logistic 时附 probability")
	format := fs.String("format", "jsonl", "jsonl|csv（默认 jsonl；csv 出单 prediction 列）")
	naPolicy := fs.String("na-policy", "error", "缺失值策略：error|skip-row")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *modelPath == "" || *dataPath == "" || *out == "" {
		return errUsage("--model/--data/--out 必需")
	}

	obj := *objective
	if obj == "" {
		if _, irObj, err := leavesio.ParseLeavesJSONFile(*modelPath); err == nil {
			obj = irObj
		}
	}

	m, err := loadEnsemble(*modelPath)
	if err != nil {
		return fmt.Errorf("load model: %w", err)
	}
	defer m.Close()

	dm, err := loadMatrix(*dataPath, *naPolicy)
	if err != nil {
		return err
	}
	vals, err := denseVals(dm)
	if err != nil {
		return err
	}

	nc := m.NOutputGroups()
	if nc < 1 {
		nc = 1
	}
	n := dm.NumRow()
	preds := make([]float64, n*nc)
	if err := m.PredictDense(vals, n, dm.NumCol(), preds, 0, 0); err != nil {
		return fmt.Errorf("predict: %w", err)
	}

	_ = os.MkdirAll(filepath.Dir(*out), 0o755)
	f, err := os.Create(*out)
	if err != nil {
		return err
	}
	defer f.Close()

	multiClass := isMulticlassObjective(obj)
	if *format == "csv" {
		return writeCSVDenseObj(f, nc, n, preds, multiClass)
	}

	enc := json.NewEncoder(f)
	for i := 0; i < n; i++ {
		row := preds[i*nc : (i+1)*nc]
		rec := map[string]any{"row": i}
		if nc == 1 {
			rec["margin"] = row[0]
			if obj == "binary:logistic" {
				rec["probability"] = sigmoid(row[0])
			}
		} else if multiClass {
			rec["class"] = argmax(row)
			rec["probabilities"] = softmaxRows(row, nc)
		} else {
			// 多目标回归（LIB-21）：输出各目标 margin，不做 softmax
			margins := make([]float64, nc)
			copy(margins, row)
			rec["margins"] = margins
			rec["predictions"] = margins
		}
		if err := enc.Encode(rec); err != nil {
			return err
		}
	}
	return nil
}

func isMulticlassObjective(obj string) bool {
	return obj == "multi:softmax" || obj == "multi:softprob"
}

// writeCSVDenseObj multiClass=true 时多列为 argmax 类；false 时输出 pred_0..pred_k-1。
func writeCSVDenseObj(w io.Writer, numOut, n int, preds []float64, multiClass bool) error {
	if numOut > 1 && multiClass {
		fmt.Fprintln(w, "prediction")
		for i := 0; i < n; i++ {
			fmt.Fprintf(w, "%d\n", argmax(preds[i*numOut:(i+1)*numOut]))
		}
		return nil
	}
	if numOut > 1 {
		// 多目标回归：多列
		for k := 0; k < numOut; k++ {
			if k > 0 {
				fmt.Fprint(w, ",")
			}
			fmt.Fprintf(w, "pred_%d", k)
		}
		fmt.Fprintln(w)
		for i := 0; i < n; i++ {
			for k := 0; k < numOut; k++ {
				if k > 0 {
					fmt.Fprint(w, ",")
				}
				fmt.Fprintf(w, "%g", preds[i*numOut+k])
			}
			fmt.Fprintln(w)
		}
		return nil
	}
	fmt.Fprintln(w, "prediction")
	for i := 0; i < n; i++ {
		fmt.Fprintf(w, "%g\n", preds[i])
	}
	return nil
}
