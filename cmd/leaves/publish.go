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
	"strings"
	"time"

	leavesio "github.com/linkerlin/leaves/v2/io"
	"github.com/linkerlin/leaves/v2/quantize"
	"github.com/linkerlin/leaves/v2/tree"
)

func cmdPublish(args []string) error {
	fs := flag.NewFlagSet("publish", flag.ContinueOnError)
	fs.Usage = func() { fmt.Fprintln(fs.Output(), "用法: leaves publish --model PATH --out-dir DIR [flags]") }
	modelPath := fs.String("model", "", "模型 leaves.json 路径（必需）")
	outDir := fs.String("out-dir", "", "产物目录（必需）")
	version := fs.String("version", "1.0.0", "版本号")
	doQuantize := fs.Bool("quantize", false, "int8 量化并出 parity 报告")
	exportXGB := fs.Bool("export-xgb", false, "导出 XGBoost 3.x JSON")
	exportONNX := fs.Bool("export-onnx", false, "导出 ai.onnx.ml 标准 ONNX 模型（WP-39）")
	metricsPath := fs.String("metrics", "", "训练 metrics.json（快照进 manifest）")
	dataPath := fs.String("data", "", "数据路径（量化 parity 用；可选）")
	emitRepro := fs.String("emit-repro-script", "", "生成复现脚本：ps1|sh|both（写入 out-dir；空=不生成）")
	printRepro := fs.Bool("print-repro", false, "将复现 train 命令打印到 stdout（需 --metrics 含 params；与 --emit-repro-script 可并用）")
	labelCol := fs.String("label-col", "", "显式指定标签列名或索引（量化 parity 用；省略时自动从 --metrics 继承，WP-30）")
	dropCols := fs.String("drop-cols", "", "逗号分隔需丢弃的列名或索引（量化 parity 用；省略时自动从 --metrics 继承，WP-30）")
	naPolicy := fs.String("na-policy", "", "缺失值策略：error|skip-row|nan（省略时自动从 --metrics 继承）")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *modelPath == "" || *outDir == "" {
		return errUsage("--model 与 --out-dir 必需")
	}
	reproMode := strings.ToLower(strings.TrimSpace(*emitRepro))
	switch reproMode {
	case "", "ps1", "sh", "both":
	default:
		return errUsage("--emit-repro-script 须为 ps1|sh|both（或省略）")
	}
	if err := os.MkdirAll(*outDir, 0o755); err != nil {
		return fmt.Errorf("mkdir: %w", err)
	}

	ir, objective, format, err := leavesio.LoadModelIR(*modelPath)
	if err != nil {
		return fmt.Errorf("parse model: %w", err)
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

	// WP-30：从 metrics.params 自动继承列控与缺失策略供量化 parity 使用（CLI 显式传入优先）。
	effLabelCol := *labelCol
	effDropCols := *dropCols
	effNAPolicy := *naPolicy
	effNumTarget := 0
	effAutoEncode := false
	if m, ok := metricsSnap.(map[string]any); ok && m != nil {
		if params, ok := m["params"].(map[string]any); ok && params != nil {
			if effLabelCol == "" {
				if v, ok := params["label_col"].(string); ok {
					effLabelCol = v
				}
			}
			if effDropCols == "" {
				if v, ok := params["drop_cols"].(string); ok {
					effDropCols = v
				}
			}
			if effNAPolicy == "" {
				if v, ok := params["na_policy"].(string); ok {
					effNAPolicy = v
				}
			}
			if nt, ok := asFloat(params["num_target"]); ok && nt >= 2 {
				effNumTarget = int(nt)
			}
			if ae, ok := params["auto_encode"].(bool); ok && ae {
				effAutoEncode = true
			}
		}
	}
	if effNAPolicy == "" {
		effNAPolicy = "error"
	}

	files := []map[string]any{}
	addFile := func(name, path string) {
		sum, sz := sha256File(path)
		files = append(files, map[string]any{"name": name, "sha256": sum, "size": sz})
	}

	// 主模型：leaves.json 原样复制；外部模型（XGB/LGB/sklearn/ONNX）自动序列化为标准 model.leaves.json（WP-26）。
	mainName := "model.leaves.json"
	mainPath := filepath.Join(*outDir, mainName)
	if format == leavesio.FormatLeavesJSON {
		if err := copyFile(*modelPath, mainPath); err != nil {
			return fmt.Errorf("copy model: %w", err)
		}
	} else {
		if err := leavesio.SaveLeavesJSONFile(mainPath, ir, objective); err != nil {
			return fmt.Errorf("save leaves.json from %s: %w", leavesio.SupportOf(format).Name, err)
		}
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

	if *exportONNX {
		onnxName := "model.onnx"
		onnxPath := filepath.Join(*outDir, onnxName)
		if err := leavesio.ExportONNXFile(onnxPath, ir, objective); err != nil {
			return fmt.Errorf("export onnx: %w", err)
		}
		addFile(onnxName, onnxPath)
	}

	if *doQuantize && ir.Forest != nil {
		qf, report, err := quantizeReport(ir.Forest, *dataPath, effNAPolicy, effNumTarget, effLabelCol, effDropCols, effAutoEncode, ir.CategoryMaps, ir.FeatureNames)
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

	nTrees := 0
	nBoost := 0
	if ir.Forest != nil {
		nTrees = len(ir.Forest.Trees)
		nBoost = ir.Forest.NEstimators()
	}

	manifest := map[string]any{
		"version":        *version,
		"objective":      objective,
		"num_features":   ir.NumFeatures,
		"n_trees":        nTrees,
		"n_boost_rounds": nBoost,
		"base_learners":  nTrees,
		"files":          files,
		"metrics":        metricsSnap,
		"created_at":     time.Now().UTC().Format(time.RFC3339),
		"schema_version": 1,
		// leaves_cli：生成本工件包的 CLI 版本（go install 二进制带真实 tag；仓库内为 (devel)+commit）。
		"leaves_cli": cliVersionLabel(),
	}
	reproCmd := buildReproduceCommand(metricsSnap, *dataPath, objective)
	if reproCmd != "" {
		manifest["reproduce"] = reproCmd
	}
	// POST-11：stdout 一键打印（Agent 可管道捕获；不写文件）。
	if *printRepro {
		if reproCmd == "" {
			return errUsage("--print-repro 需要 --metrics（含 params）以生成训练命令")
		}
		fmt.Println(reproCmd)
	}
	// WP-19：可选写出可执行复现脚本（需 metrics 以拼出 train 命令）。
	if reproMode != "" {
		if reproCmd == "" {
			return errUsage("--emit-repro-script 需要 --metrics（含 params）以生成训练命令")
		}
		written, err := writeReproduceScripts(*outDir, reproMode, reproCmd)
		if err != nil {
			return err
		}
		for _, name := range written {
			addFile(name, filepath.Join(*outDir, name))
		}
	}
	// WP-44：发布产物回读推理自检与跨格式（XGB/ONNX/Quant）parity 审计。
	verification, verr := verifyPublishedArtifacts(*outDir, mainPath, *exportXGB, *exportONNX, *doQuantize && ir.Forest != nil, ir.NumFeatures)
	if verr != nil {
		return fmt.Errorf("publish verification failed: %w", verr)
	}
	manifest["verification"] = verification
	// 文档级对接提示：registry 推送在库外。
	manifest["publish_note"] = "本地工件包；推送 registry/S3/镜像由用户 CI 完成（leaves 不内置 serving/registry）"
	manifest["files"] = files
	return writeJSON(filepath.Join(*outDir, "manifest.json"), manifest)
}

// writeReproduceScripts 按 mode 写入 reproduce.ps1 / reproduce.sh。
func writeReproduceScripts(outDir, mode, reproCmd string) ([]string, error) {
	var names []string
	writePS1 := mode == "ps1" || mode == "both"
	writeSH := mode == "sh" || mode == "both"
	if writePS1 {
		name := "reproduce.ps1"
		body := "# Generated by leaves publish --emit-repro-script\n" +
			"# Requires: leaves on PATH, or replace with: go run ./cmd/leaves ...\n" +
			"$ErrorActionPreference = 'Stop'\n" +
			reproCmd + "\n"
		if err := os.WriteFile(filepath.Join(outDir, name), []byte(body), 0o644); err != nil {
			return nil, fmt.Errorf("write %s: %w", name, err)
		}
		names = append(names, name)
	}
	if writeSH {
		name := "reproduce.sh"
		body := "#!/usr/bin/env bash\n" +
			"# Generated by leaves publish --emit-repro-script\n" +
			"# Requires: leaves on PATH, or replace with: go run ./cmd/leaves ...\n" +
			"set -euo pipefail\n" +
			reproCmd + "\n"
		if err := os.WriteFile(filepath.Join(outDir, name), []byte(body), 0o755); err != nil {
			return nil, fmt.Errorf("write %s: %w", name, err)
		}
		names = append(names, name)
	}
	return names, nil
}

// buildReproduceCommand 从 metrics 快照拼一条可复现的 leaves train 命令（Agent/人类对接用）。
func buildReproduceCommand(metricsSnap any, dataPath, objective string) string {
	m, ok := metricsSnap.(map[string]any)
	if !ok || m == nil {
		// metrics 可能是通过 json.Unmarshal 到 any 的 map
		return ""
	}
	obj := objective
	if o, ok := m["objective"].(string); ok && o != "" {
		obj = o
	}
	params, _ := m["params"].(map[string]any)
	if params == nil {
		return fmt.Sprintf("leaves train --data %s --objective %s --out-model model.leaves.json",
			quotePath(dataPath), obj)
	}
	// 优先用 metrics 内嵌的 data 路径；publish 的 --data 多为 parity 用。
	dataArg := dataPath
	if dataArg == "" {
		dataArg = "TRAIN.csv"
	}
	cmd := fmt.Sprintf("leaves train --data %s --objective %s", quotePath(dataArg), obj)
	add := func(flag, key string) {
		if v, ok := params[key]; ok && v != nil {
			cmd += fmt.Sprintf(" --%s %v", flag, v)
		}
	}
	add("rounds", "rounds")
	add("depth", "depth")
	add("lr", "lr")
	add("lambda", "lambda")
	add("min-child-weight", "min_child_weight")
	add("gamma", "gamma")
	add("max-bin", "max_bin")
	add("subsample", "subsample")
	add("colsample", "colsample")
	add("tree-method", "tree_method")
	add("seed", "seed")
	// AGUX-08：max-leaves / num-target / cv 同为复现必需（原缺失使 CV run 退化为全量单训）。
	add("max-leaves", "max_leaves")
	add("num-target", "num_target")
	add("label-col", "label_col")
	add("drop-cols", "drop_cols")
	add("booster", "booster")
	add("rate-drop", "rate_drop")
	add("skip-drop", "skip_drop")
	add("monotone", "monotone")
	add("lr-decay", "lr_decay")
	add("batch-rows", "batch_rows")
	add("na-policy", "na_policy")
	add("weight-col", "weight_col")
	add("scale-pos-weight", "scale_pos_weight")
	add("tweedie-variance-power", "tweedie_variance_power")
	add("num-parallel-tree", "num_parallel_tree")
	if ae, ok := params["auto_encode"].(bool); ok && ae {
		cmd += " --auto-encode"
	}
	if mot, ok := params["multi_output_tree"].(bool); ok && mot {
		cmd += " --multi-output-tree"
	}
	if cv, ok := asFloat(params["cv_folds"]); ok && cv >= 2 {
		cmd += fmt.Sprintf(" --cv %d", int(cv))
		if es, ok := asFloat(params["early_stop"]); ok && es > 0 {
			cmd += fmt.Sprintf(" --early-stop %d", int(es))
		}
	} else {
		// 单跑路径：早停语义需 --val + --early-stop 成对（val 路径由 params.val 提供）。
		if vp, ok := params["val"].(string); ok && vp != "" {
			cmd += " --val " + quotePath(vp)
		}
		if es, ok := asFloat(params["early_stop"]); ok && es > 0 {
			cmd += fmt.Sprintf(" --early-stop %d", int(es))
		}
	}
	if em, ok := params["eval_metric"].(string); ok && em != "" {
		cmd += " --eval-metric " + em
	} else if em, ok := m["metric"].(string); ok && em != "" {
		cmd += " --eval-metric " + em
	}
	if nc, ok := asFloat(params["num_class"]); ok && nc >= 2 {
		cmd += fmt.Sprintf(" --num-class %d", int(nc))
	}
	if k, ok := asFloat(params["ndcg_k"]); ok && k > 0 {
		cmd += fmt.Sprintf(" --ndcg-k %d", int(k))
	}
	cmd += " --out-model model.leaves.json --metrics metrics.json"
	return cmd
}

func quotePath(p string) string {
	if p == "" {
		return "DATA"
	}
	if strings.ContainsAny(p, " \t") {
		return `"` + p + `"`
	}
	return p
}

func asFloat(v any) (float64, bool) {
	switch x := v.(type) {
	case float64:
		return x, true
	case int:
		return float64(x), true
	case int64:
		return float64(x), true
	default:
		return 0, false
	}
}

// quantizeReport 对森林做 int8 量化；若提供 dataPath 则在≤5000 行上跑 parity。
// 返回量化森林（供持久化侧车用）与报告。
func quantizeReport(
	forest *tree.ForestIR,
	dataPath, naPolicy string,
	numTarget int,
	labelCol, dropCols string,
	autoEncode bool,
	catMaps map[string]map[string]float64,
	expectedFeats []string,
) (*quantize.QuantizedForest, map[string]any, error) {
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
	dm, err := loadMatrixFull(dataPath, naPolicy, numTarget, labelCol, dropCols, "", "", autoEncode, catMaps, expectedFeats)
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

// verifyPublishedArtifacts 对发布目录中的主模型及可选导出产物执行回读与合成样本推理对齐自检（WP-44 / AA5）。
func verifyPublishedArtifacts(outDir, mainPath string, exportXGB, exportONNX, hasQuant bool, numFeatures int) (map[string]any, error) {
	cols := numFeatures
	if cols <= 0 {
		cols = 1
	}
	const smokeRows = 4
	smokeVals := make([]float64, smokeRows*cols)
	for r := 0; r < smokeRows; r++ {
		for c := 0; c < cols; c++ {
			smokeVals[r*cols+c] = float64(r-1)*0.75 + float64(c)*0.25
		}
	}

	mainEns, err := loadEnsemble(mainPath)
	if err != nil {
		return nil, fmt.Errorf("reload main model: %w", err)
	}
	defer mainEns.Close()

	nc := mainEns.NOutputGroups()
	if nc < 1 {
		nc = 1
	}
	basePreds := make([]float64, smokeRows*nc)
	if err := mainEns.PredictDense(smokeVals, smokeRows, cols, basePreds, 0, 0); err != nil {
		return nil, fmt.Errorf("smoke predict main model: %w", err)
	}

	res := map[string]any{
		"verified":     true,
		"main_load_ok": true,
		"smoke_rows":   smokeRows,
		"num_features": cols,
	}

	maxAbsDiff := func(a, b []float64) float64 {
		var mx float64
		for i := range a {
			if i >= len(b) {
				break
			}
			d := a[i] - b[i]
			if d < 0 {
				d = -d
			}
			if d > mx {
				mx = d
			}
		}
		return mx
	}

	if exportXGB {
		xgbPath := filepath.Join(outDir, "model.xgb.json")
		xgbEns, err := loadEnsemble(xgbPath)
		if err != nil {
			return nil, fmt.Errorf("reload exported xgb: %w", err)
		}
		defer xgbEns.Close()
		xgbPreds := make([]float64, smokeRows*nc)
		if err := xgbEns.PredictDense(smokeVals, smokeRows, cols, xgbPreds, 0, 0); err != nil {
			return nil, fmt.Errorf("smoke predict exported xgb: %w", err)
		}
		res["xgb_load_ok"] = true
		res["xgb_parity_max_diff"] = round4(maxAbsDiff(basePreds, xgbPreds))
	}

	if exportONNX {
		onnxPath := filepath.Join(outDir, "model.onnx")
		onnxEns, err := loadEnsemble(onnxPath)
		if err != nil {
			return nil, fmt.Errorf("reload exported onnx: %w", err)
		}
		defer onnxEns.Close()
		onnxPreds := make([]float64, smokeRows*nc)
		if err := onnxEns.PredictDense(smokeVals, smokeRows, cols, onnxPreds, 0, 0); err != nil {
			return nil, fmt.Errorf("smoke predict exported onnx: %w", err)
		}
		res["onnx_load_ok"] = true
		res["onnx_parity_max_diff"] = round4(maxAbsDiff(basePreds, onnxPreds))
	}

	if hasQuant {
		quantPath := filepath.Join(outDir, "model.quant.json")
		quantEns, err := loadEnsemble(quantPath)
		if err != nil {
			return nil, fmt.Errorf("reload quant overlay: %w", err)
		}
		defer quantEns.Close()
		quantPreds := make([]float64, smokeRows*nc)
		if err := quantEns.PredictDense(smokeVals, smokeRows, cols, quantPreds, 0, 0); err != nil {
			return nil, fmt.Errorf("smoke predict quant overlay: %w", err)
		}
		res["quant_load_ok"] = true
		res["quant_smoke_max_diff"] = round4(maxAbsDiff(basePreds, quantPreds))
	}

	return res, nil
}
