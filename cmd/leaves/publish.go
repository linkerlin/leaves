package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"time"

	"github.com/linkerlin/leaves/data"
	leavesio "github.com/linkerlin/leaves/io"
	"github.com/linkerlin/leaves/quantize"
	"github.com/linkerlin/leaves/tree"
)

func cmdPublish(args []string) error {
	fs := flag.NewFlagSet("publish", flag.ContinueOnError)
	fs.Usage = func() { fmt.Fprintln(fs.Output(), "用法: leaves publish --model PATH --out-dir DIR [flags]") }
	modelPath := fs.String("model", "", "模型 leaves.json 路径（必需）")
	outDir := fs.String("out-dir", "", "产物目录（必需）")
	version := fs.String("version", "1.0.0", "版本号")
	doQuantize := fs.Bool("quantize", false, "int8 量化并出 parity 报告")
	exportXGB := fs.Bool("export-xgb", false, "导出 XGBoost 3.x JSON")
	metricsPath := fs.String("metrics", "", "训练 metrics.json（快照进 manifest）")
	dataPath := fs.String("data", "", "数据路径（量化 parity 用；可选）")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *modelPath == "" || *outDir == "" {
		return errUsage("--model 与 --out-dir 必需")
	}
	if err := os.MkdirAll(*outDir, 0o755); err != nil {
		return fmt.Errorf("mkdir: %w", err)
	}

	ir, objective, err := leavesio.ParseLeavesJSONFile(*modelPath)
	if err != nil {
		return fmt.Errorf("parse model: %w", err)
	}

	files := []map[string]any{}
	addFile := func(name, path string) {
		sum, sz := sha256File(path)
		files = append(files, map[string]any{"name": name, "sha256": sum, "size": sz})
	}

	// 主模型：原样复制。
	mainName := "model.leaves.json"
	mainPath := filepath.Join(*outDir, mainName)
	if err := copyFile(*modelPath, mainPath); err != nil {
		return fmt.Errorf("copy model: %w", err)
	}
	addFile(mainName, mainPath)

	if *exportXGB {
		xgbName := "model.xgb.json"
		xgbPath := filepath.Join(*outDir, xgbName)
		if err := leavesio.ExportXGBoostJSONFile(xgbPath, ir, objective); err != nil {
			return fmt.Errorf("export xgb: %w", err)
		}
		addFile(xgbName, xgbPath)
	}

	if *doQuantize && ir.Forest != nil {
		qf, report, err := quantizeReport(ir.Forest, *dataPath)
		if err != nil {
			return fmt.Errorf("quantize: %w", err)
		}
		// 持久化量化侧车：base + overlay 即可重建量化推理（见 model_load.go）。
		quantName := "model.quant.json"
		quantPath := filepath.Join(*outDir, quantName)
		if err := quantize.SaveOverlayFile(quantPath, "model.leaves.json", qf); err != nil {
			return fmt.Errorf("save quant overlay: %w", err)
		}
		addFile(quantName, quantPath)

		reportPath := filepath.Join(*outDir, "quantize_report.json")
		if err := writeJSON(reportPath, report); err != nil {
			return err
		}
		addFile("quantize_report.json", reportPath)
	}

	var metricsSnap any
	if *metricsPath != "" {
		if b, err := os.ReadFile(*metricsPath); err == nil {
			var mv any
			if json.Unmarshal(b, &mv) == nil {
				metricsSnap = mv
			}
		}
	}

	nTrees := 0
	if ir.Forest != nil {
		nTrees = len(ir.Forest.Trees)
	}

	manifest := map[string]any{
		"version":       *version,
		"objective":     objective,
		"num_features":  ir.NumFeatures,
		"n_trees":       nTrees,
		"base_learners": nTrees,
		"files":         files,
		"metrics":       metricsSnap,
		"created_at":    time.Now().UTC().Format(time.RFC3339),
	}
	return writeJSON(filepath.Join(*outDir, "manifest.json"), manifest)
}

// quantizeReport 对森林做 int8 量化；若提供 dataPath 则在≤5000 行上跑 parity。
// 返回量化森林（供持久化侧车用）与报告。
func quantizeReport(forest *tree.ForestIR, dataPath string) (*quantize.QuantizedForest, map[string]any, error) {
	qf, err := quantize.QuantizeForest(forest, quantize.Config{})
	if err != nil {
		return nil, nil, err
	}
	report := map[string]any{
		"levels":            quantize.Levels,
		"num_features":      forest.NumFeatures,
		"max_threshold_err": qf.MaxThresholdQuantError(),
		"parity":            nil,
		"parity_note":       "未提供 --data，跳过 parity",
	}
	if dataPath == "" {
		return qf, report, nil
	}
	dm, err := data.FromFileAuto(dataPath)
	if err != nil {
		return qf, report, nil // 数据读不出则降级，不阻断发布。
	}
	vals, err := denseVals(dm)
	if err != nil {
		return qf, report, nil
	}
	cols := dm.NumCol()
	if cols <= 0 {
		return qf, report, nil
	}
	n := len(vals) / cols
	if n > 5000 {
		n = 5000 // ponytail: 发布阶段 parity 取样上限，避免跑全量。
	}
	rows := make([][]float64, n)
	for i := 0; i < n; i++ {
		rows[i] = vals[i*cols : (i+1)*cols]
	}
	gate := quantize.DefaultGate()
	res, err := quantize.CheckParityWithGate(forest, qf, rows, 0, gate)
	if err != nil {
		return qf, report, nil
	}
	delete(report, "parity_note")
	report["parity"] = map[string]any{
		"samples":          res.Samples,
		"max_margin_diff":  res.MaxMarginDiff,
		"mean_margin_diff": res.MeanMarginDiff,
		"failures":         res.Failures,
		"pass":             res.Pass(gate),
	}
	return qf, report, nil
}

func copyFile(src, dst string) error {
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()
	out, err := os.Create(dst)
	if err != nil {
		return err
	}
	defer out.Close()
	_, err = io.Copy(out, in)
	return err
}

func sha256File(path string) (string, int64) {
	f, err := os.Open(path)
	if err != nil {
		return "", 0
	}
	defer f.Close()
	h := sha256.New()
	n, _ := io.Copy(h, f)
	return hex.EncodeToString(h.Sum(nil)), n
}

func writeJSON(path string, v any) error {
	b, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return err
	}
	b = append(b, '\n')
	if path == "" || path == "-" {
		_, err = os.Stdout.Write(b)
		return err
	}
	_ = os.MkdirAll(filepath.Dir(path), 0o755)
	return os.WriteFile(path, b, 0o644)
}
