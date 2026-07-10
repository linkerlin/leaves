package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

// TestEndToEndSmoke 端到端验证 train→eval→predict→publish 全链路（ponytail: 唯一自检）。
func TestEndToEndSmoke(t *testing.T) {
	dir := t.TempDir()

	// 2 特征 + 末列 label：y = x0；满足嗅探「≥3 列、全数值」→ label-last。
	var b strings.Builder
	for i := 0; i < 16; i++ {
		fi := float64(i)
		b.WriteString(itoa2(fi) + "," + itoa2(fi*2) + "," + itoa2(fi) + "\n")
	}
	trainPath := filepath.Join(dir, "train.csv")
	if err := os.WriteFile(trainPath, []byte(b.String()), 0o644); err != nil {
		t.Fatal(err)
	}
	modelPath := filepath.Join(dir, "m.leaves.json")
	metricsPath := filepath.Join(dir, "metrics.json")

	if err := cmdTrain([]string{
		"--data", trainPath, "--objective", "reg:squarederror",
		"--eval-metric", "rmse", "--rounds", "20", "--depth", "3", "--lr", "0.3",
		"--out-model", modelPath, "--metrics", metricsPath,
	}); err != nil {
		t.Fatalf("train: %v", err)
	}

	raw, err := os.ReadFile(metricsPath)
	if err != nil {
		t.Fatalf("read metrics: %v", err)
	}
	var doc metricsDoc
	if err := json.Unmarshal(raw, &doc); err != nil {
		t.Fatalf("metrics json: %v", err)
	}
	if doc.Metric != "rmse" || doc.Maximize {
		t.Fatalf("metrics doc wrong: %+v", doc)
	}

	evalPath := filepath.Join(dir, "eval.json")
	if err := cmdEval([]string{"--model", modelPath, "--data", trainPath, "--metrics", evalPath}); err != nil {
		t.Fatalf("eval: %v", err)
	}

	predPath := filepath.Join(dir, "pred.jsonl")
	if err := cmdPredict([]string{"--model", modelPath, "--data", trainPath, "--out", predPath}); err != nil {
		t.Fatalf("predict: %v", err)
	}
	predBytes, err := os.ReadFile(predPath)
	if err != nil || !strings.Contains(string(predBytes), "\"margin\"") {
		t.Fatalf("predict out missing margin: %v", err)
	}

	relDir := filepath.Join(dir, "release")
	if err := cmdPublish([]string{
		"--model", modelPath, "--out-dir", relDir, "--version", "0.1",
		"--export-xgb", "--quantize", "--data", trainPath, "--metrics", metricsPath,
	}); err != nil {
		t.Fatalf("publish: %v", err)
	}
	man, err := os.ReadFile(filepath.Join(relDir, "manifest.json"))
	if err != nil {
		t.Fatalf("manifest: %v", err)
	}
	if !strings.Contains(string(man), "sha256") || !strings.Contains(string(man), "n_trees") {
		t.Fatalf("manifest missing fields: %s", man)
	}
}

func itoa2(f float64) string {
	return strconv.FormatFloat(f, 'f', -1, 64)
}
