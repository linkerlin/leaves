package main

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

// 写一个带表头、末列 label 的 CSV（满足嗅探 label-last）。
func writeCSV(t *testing.T, path, header string, rows []string) {
	t.Helper()
	content := header + "\n" + strings.Join(rows, "\n") + "\n"
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
}

func f2(v float64) string { return strconv.FormatFloat(v, 'f', -1, 64) }

// TestBinaryEndToEnd 验证二分类决策表路径：binary:logistic + logloss + probability 输出。
func TestBinaryEndToEnd(t *testing.T) {
	dir := t.TempDir()
	trainPath := filepath.Join(dir, "bin.csv")
	var rows []string
	for i := 0; i < 40; i++ {
		x0 := float64(i) * 0.1
		x1 := float64(i%5) * 0.3
		y := 0.0
		if 2*x0+x1 > 1.0 {
			y = 1.0
		}
		rows = append(rows, f2(x0)+","+f2(x1)+","+f2(y))
	}
	writeCSV(t, trainPath, "x0,x1,label", rows)

	modelPath := filepath.Join(dir, "bin.leaves.json")
	metricsPath := filepath.Join(dir, "bin.json")
	if err := cmdTrain([]string{
		"--data", trainPath, "--objective", "binary:logistic",
		"--eval-metric", "logloss", "--rounds", "30", "--depth", "3", "--lr", "0.3",
		"--out-model", modelPath, "--metrics", metricsPath,
	}); err != nil {
		t.Fatalf("train: %v", err)
	}
	var doc metricsDoc
	b, _ := os.ReadFile(metricsPath)
	if err := json.Unmarshal(b, &doc); err != nil {
		t.Fatalf("metrics: %v", err)
	}
	if doc.Metric != "logloss" || doc.Maximize {
		t.Fatalf("binary metrics wrong: %+v", doc)
	}

	predPath := filepath.Join(dir, "bin.jsonl")
	if err := cmdPredict([]string{
		"--model", modelPath, "--data", trainPath, "--out", predPath,
		"--objective", "binary:logistic",
	}); err != nil {
		t.Fatalf("predict: %v", err)
	}
	pred, _ := os.ReadFile(predPath)
	if !strings.Contains(string(pred), "probability") {
		t.Fatalf("binary predict missing probability: %s", pred)
	}
}

// TestRankingEndToEnd 验证排序决策表路径：rank:ndcg + ndcg@10 + group 评估。
func TestRankingEndToEnd(t *testing.T) {
	dir := t.TempDir()
	trainPath := filepath.Join(dir, "rank.tsv")
	// 12 个 qid，每组 4 行；label=relevance，feat0=rel（强信号），feat1 噪声。
	var lines []string
	for qid := 0; qid < 12; qid++ {
		for _, rel := range []int{3, 2, 1, 0} {
			lines = append(lines, strconv.Itoa(qid)+"\t"+strconv.Itoa(rel)+
				"\t"+strconv.Itoa(rel)+"\t"+f2(float64(rel)*0.5))
		}
	}
	if err := os.WriteFile(trainPath, []byte(strings.Join(lines, "\n")+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	modelPath := filepath.Join(dir, "rank.leaves.json")
	metricsPath := filepath.Join(dir, "rank.json")
	if err := cmdTrain([]string{
		"--data", trainPath, "--objective", "rank:ndcg",
		"--eval-metric", "ndcg@10", "--ndcg-k", "10",
		"--rounds", "30", "--depth", "3", "--lr", "0.3",
		"--out-model", modelPath, "--metrics", metricsPath,
	}); err != nil {
		t.Fatalf("train: %v", err)
	}
	var doc metricsDoc
	b, _ := os.ReadFile(metricsPath)
	if err := json.Unmarshal(b, &doc); err != nil {
		t.Fatalf("metrics: %v", err)
	}
	if !doc.Maximize {
		t.Fatalf("ndcg should be maximize: %+v", doc)
	}

	// eval 需 group：eval 内部用 groupsOf(dm) 取 DenseWithGroups 的组。
	evalPath := filepath.Join(dir, "rank_eval.json")
	if err := cmdEval([]string{
		"--model", modelPath, "--data", trainPath,
		"--eval-metric", "ndcg@10", "--metrics", evalPath,
	}); err != nil {
		t.Fatalf("eval: %v", err)
	}
}

// TestMulticlassEndToEnd 验证多分类决策表路径：multi:softmax + mlogloss + 概率向量。
func TestMulticlassEndToEnd(t *testing.T) {
	dir := t.TempDir()
	trainPath := filepath.Join(dir, "multi.csv")
	var rows []string
	for i := 0; i < 80; i++ {
		c := float64(i % 4) // 类 0..3，特征与类强相关
		rows = append(rows, f2(c)+","+f2(c*0.5+0.1)+","+f2(c*0.3)+","+f2(c))
	}
	writeCSV(t, trainPath, "f0,f1,f2,label", rows)

	modelPath := filepath.Join(dir, "multi.leaves.json")
	metricsPath := filepath.Join(dir, "multi.json")
	if err := cmdTrain([]string{
		"--data", trainPath, "--objective", "multi:softmax",
		"--num-class", "4", "--eval-metric", "mlogloss",
		"--rounds", "30", "--depth", "3", "--lr", "0.3",
		"--out-model", modelPath, "--metrics", metricsPath,
	}); err != nil {
		t.Fatalf("train: %v", err)
	}
	var doc metricsDoc
	b, _ := os.ReadFile(metricsPath)
	if err := json.Unmarshal(b, &doc); err != nil {
		t.Fatalf("metrics: %v", err)
	}
	if doc.Metric != "mlogloss" || doc.Maximize {
		t.Fatalf("multiclass metrics wrong: %+v", doc)
	}

	predPath := filepath.Join(dir, "multi.jsonl")
	if err := cmdPredict([]string{"--model", modelPath, "--data", trainPath, "--out", predPath}); err != nil {
		t.Fatalf("predict: %v", err)
	}
	pred, _ := os.ReadFile(predPath)
	if !strings.Contains(string(pred), "class") || !strings.Contains(string(pred), "probabilities") {
		t.Fatalf("multiclass predict missing class/probabilities: %s", pred)
	}

	evalPath := filepath.Join(dir, "multi_eval.json")
	if err := cmdEval([]string{
		"--model", modelPath, "--data", trainPath,
		"--eval-metric", "mlogloss", "--metrics", evalPath,
	}); err != nil {
		t.Fatalf("eval: %v", err)
	}
}

// TestSniff 验证数据画像自动推荐 objective/metric（闭环第一步无需人工告知任务类型）。
func TestSniff(t *testing.T) {
	dir := t.TempDir()

	// 二分类：label ∈ {0,1} → binary:logistic
	binPath := filepath.Join(dir, "b.csv")
	writeCSV(t, binPath, "x0,x1,label", []string{"0.1,0.2,0", "0.3,0.4,1", "0.5,0.6,0", "0.7,0.8,1"})
	checkSniff(t, binPath, "binary:logistic")

	// 回归：label 连续 → reg:squarederror
	regPath := filepath.Join(dir, "r.csv")
	writeCSV(t, regPath, "x0,label", []string{"0.1,3.14", "0.2,2.71", "0.3,0.58"})
	checkSniff(t, regPath, "reg:squarederror")

	// 排序：qid label feat → rank:ndcg
	rankPath := filepath.Join(dir, "rk.tsv")
	rankLines := []string{}
	for qid := 0; qid < 3; qid++ {
		for _, rel := range []int{3, 1, 0} {
			rankLines = append(rankLines, strconv.Itoa(qid)+"\t"+strconv.Itoa(rel)+"\t0.5\t0.6")
		}
	}
	os.WriteFile(rankPath, []byte(strings.Join(rankLines, "\n")+"\n"), 0o644)
	checkSniff(t, rankPath, "rank:ndcg")
}

func checkSniff(t *testing.T, path, wantObj string) {
	t.Helper()
	out := path + ".sniff.json"
	if err := cmdSniff([]string{"--data", path, "--metrics", out}); err != nil {
		t.Fatalf("sniff %s: %v", path, err)
	}
	b, err := os.ReadFile(out)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(b), `"suggested_objective": "`+wantObj+`"`) {
		t.Fatalf("sniff %s: expected %s, got %s", path, wantObj, b)
	}
}

// TestEmitRounds 验证 --emit-rounds 输出逐轮指标 JSONL（Agent 学习曲线诊断）。
// TestExplainImportance 验证特征重要性（Agent 诊断"模型为什么这样预测"）。
func TestExplainImportance(t *testing.T) {
	dir := t.TempDir()
	trainPath := filepath.Join(dir, "t.csv")
	writeCSV(t, trainPath, "x0,x1,x2,label", []string{"0.1,0.2,0.3,0.5", "0.4,0.5,0.6,1.2", "0.7,0.8,0.9,2.0"})
	modelPath := filepath.Join(dir, "m.leaves.json")
	if err := cmdTrain([]string{"--data", trainPath, "--objective", "reg:squarederror", "--rounds", "5", "--depth", "2", "--out-model", modelPath}); err != nil {
		t.Fatalf("train: %v", err)
	}
	out := filepath.Join(dir, "imp.json")
	if err := cmdExplain([]string{"--model", modelPath, "--type", "importance", "--metrics", out}); err != nil {
		t.Fatalf("explain: %v", err)
	}
	b, _ := os.ReadFile(out)
	if !strings.Contains(string(b), "features") || !strings.Contains(string(b), "n_features") {
		t.Fatalf("importance missing features: %s", b)
	}
}

// TestPredictCSV 验证 CSV 部署输出格式。
func TestPredictCSV(t *testing.T) {
	dir := t.TempDir()
	trainPath := filepath.Join(dir, "t.csv")
	var rows []string
	for i := 0; i < 20; i++ {
		fi := float64(i)
		rows = append(rows, f2(fi)+","+f2(fi*0.5)+","+f2(fi*2))
	}
	writeCSV(t, trainPath, "x0,x1,label", rows)
	modelPath := filepath.Join(dir, "m.leaves.json")
	if err := cmdTrain([]string{"--data", trainPath, "--objective", "reg:squarederror", "--rounds", "5", "--depth", "2", "--out-model", modelPath}); err != nil {
		t.Fatalf("train: %v", err)
	}
	out := filepath.Join(dir, "out.csv")
	if err := cmdPredict([]string{"--model", modelPath, "--data", trainPath, "--out", out, "--format", "csv"}); err != nil {
		t.Fatalf("predict: %v", err)
	}
	b, _ := os.ReadFile(out)
	if !strings.Contains(string(b), "prediction") {
		t.Fatalf("csv missing header: %s", b)
	}
	lines := strings.Split(strings.TrimSpace(string(b)), "\n")
	if len(lines) != 21 { // header + 20 rows
		t.Fatalf("expected 21 lines, got %d", len(lines))
	}
}

func TestEmitRounds(t *testing.T) {
	dir := t.TempDir()
	trainPath := filepath.Join(dir, "t.csv")
	var rows []string
	for i := 0; i < 30; i++ {
		fi := float64(i)
		rows = append(rows, f2(fi)+","+f2(fi*0.5)+","+f2(fi*1.5))
	}
	writeCSV(t, trainPath, "x0,x1,label", rows)
	roundsPath := filepath.Join(dir, "rounds.jsonl")
	if err := cmdTrain([]string{
		"--data", trainPath, "--objective", "reg:squarederror",
		"--rounds", "10", "--depth", "2",
		"--emit-rounds", roundsPath,
	}); err != nil {
		t.Fatalf("train: %v", err)
	}
	b, _ := os.ReadFile(roundsPath)
	lines := strings.Split(strings.TrimSpace(string(b)), "\n")
	if len(lines) != 10 {
		t.Fatalf("expected 10 round lines, got %d", len(lines))
	}
	var last map[string]any
	if err := json.Unmarshal([]byte(lines[9]), &last); err != nil {
		t.Fatal(err)
	}
	if last["round"].(float64) != 9 || last["train"] == nil {
		t.Fatalf("unexpected last round: %+v", last)
	}
}

// TestAutoObjective 验证 eval/predict 自动从模型推断 objective（≡正确默认 metric/probability）。
func TestAutoObjective(t *testing.T) {
	dir := t.TempDir()

	// 二分类：train binary:logistic → eval 无 --eval-metric 自动 logloss → predict 无 --objective 自动 probability
	binPath := filepath.Join(dir, "b.csv")
	writeCSV(t, binPath, "x0,x1,label", []string{"0.1,0.2,0", "0.3,0.4,1", "0.5,0.6,0", "0.7,0.8,1", "0.9,1.0,1", "1.1,1.2,1"})
	bm := filepath.Join(dir, "b.leaves.json")
	if err := cmdTrain([]string{"--data", binPath, "--objective", "binary:logistic", "--rounds", "5", "--depth", "2", "--out-model", bm}); err != nil {
		t.Fatalf("train: %v", err)
	}
	eOut := filepath.Join(dir, "beval.json")
	if err := cmdEval([]string{"--model", bm, "--data", binPath, "--metrics", eOut}); err != nil {
		t.Fatalf("eval: %v", err)
	}
	b, _ := os.ReadFile(eOut)
	if !strings.Contains(string(b), `"metric": "logloss"`) {
		t.Fatalf("binary eval should auto-logloss: %s", b)
	}
	predOut := filepath.Join(dir, "bp.jsonl")
	if err := cmdPredict([]string{"--model", bm, "--data", binPath, "--out", predOut}); err != nil {
		t.Fatalf("predict: %v", err)
	}
	pred, _ := os.ReadFile(predOut)
	if !strings.Contains(string(pred), "probability") {
		t.Fatalf("binary predict should auto-probability: %s", pred)
	}

	// 排序：train rank:ndcg → eval 无 --eval-metric 自动 ndcg@10
	rankPath := filepath.Join(dir, "r.tsv")
	var rl []string
	for q := 0; q < 4; q++ {
		for _, rel := range []int{3, 1, 0} {
			rl = append(rl, strconv.Itoa(q)+"\t"+strconv.Itoa(rel)+"\t0.5\t0.6")
		}
	}
	os.WriteFile(rankPath, []byte(strings.Join(rl, "\n")+"\n"), 0o644)
	rm := filepath.Join(dir, "r.leaves.json")
	if err := cmdTrain([]string{"--data", rankPath, "--objective", "rank:ndcg", "--ndcg-k", "10", "--rounds", "5", "--depth", "2", "--out-model", rm}); err != nil {
		t.Fatalf("train: %v", err)
	}
	reOut := filepath.Join(dir, "reval.json")
	if err := cmdEval([]string{"--model", rm, "--data", rankPath, "--metrics", reOut}); err != nil {
		t.Fatalf("eval: %v", err)
	}
	b, _ = os.ReadFile(reOut)
	if !strings.Contains(string(b), `"metric": "ndcg@10"`) || !strings.Contains(string(b), `"maximize": true`) {
		t.Fatalf("ranking eval should auto-ndcg@10 + maximize: %s", b)
	}
}

func parseMargins(t *testing.T, path string) []float64 {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var out []float64
	for _, line := range strings.Split(strings.TrimSpace(string(b)), "\n") {
		if line == "" {
			continue
		}
		var rec struct {
			Margin float64 `json:"margin"`
		}
		if json.Unmarshal([]byte(line), &rec) == nil {
			out = append(out, rec.Margin)
		}
	}
	return out
}

// TestQuantizeRoundTrip 验证 publish --quantize 持久化的侧车可被 predict 重建，
// 且量化 margin 与原模型在 parity 门禁（0.15）内一致。
func TestQuantizeRoundTrip(t *testing.T) {
	dir := t.TempDir()
	trainPath := filepath.Join(dir, "train.csv")
	var rows []string
	for i := 0; i < 24; i++ {
		fi := float64(i)
		rows = append(rows, f2(fi)+","+f2(fi*0.5)+","+f2(fi*1.5))
	}
	writeCSV(t, trainPath, "x0,x1,label", rows)

	modelPath := filepath.Join(dir, "m.leaves.json")
	if err := cmdTrain([]string{
		"--data", trainPath, "--objective", "reg:squarederror",
		"--rounds", "20", "--depth", "4", "--out-model", modelPath,
	}); err != nil {
		t.Fatalf("train: %v", err)
	}
	relDir := filepath.Join(dir, "release")
	if err := cmdPublish([]string{
		"--model", modelPath, "--out-dir", relDir, "--quantize", "--data", trainPath,
	}); err != nil {
		t.Fatalf("publish: %v", err)
	}
	quantModel := filepath.Join(relDir, "model.quant.json")
	if _, err := os.Stat(quantModel); err != nil {
		t.Fatalf("quant overlay not produced: %v", err)
	}

	basePred := filepath.Join(dir, "base.jsonl")
	quantPred := filepath.Join(dir, "quant.jsonl")
	if err := cmdPredict([]string{"--model", modelPath, "--data", trainPath, "--out", basePred}); err != nil {
		t.Fatalf("predict base: %v", err)
	}
	if err := cmdPredict([]string{"--model", quantModel, "--data", trainPath, "--out", quantPred}); err != nil {
		t.Fatalf("predict quant: %v", err)
	}
	bm := parseMargins(t, basePred)
	qm := parseMargins(t, quantPred)
	if len(bm) != len(qm) {
		t.Fatalf("margin count mismatch: %d vs %d", len(bm), len(qm))
	}
	maxDiff := 0.0
	for i := range bm {
		if d := math.Abs(bm[i] - qm[i]); d > maxDiff {
			maxDiff = d
		}
	}
	if maxDiff > 0.15 { // quantize.DefaultGate().MaxMarginDiff
		t.Fatalf("quant round-trip max margin diff %g exceeds parity gate", maxDiff)
	}
}

func TestRunsLedger(t *testing.T) {
	dir := t.TempDir()
	trainPath := filepath.Join(dir, "r.csv")
	var rows []string
	for i := 0; i < 16; i++ {
		fi := float64(i)
		rows = append(rows, f2(fi)+","+f2(fi*2)+","+f2(fi))
	}
	writeCSV(t, trainPath, "x0,x1,label", rows)

	runs := filepath.Join(dir, "runs.jsonl")
	for _, tag := range []string{"baseline", "tune1"} {
		if err := cmdTrain([]string{
			"--data", trainPath, "--objective", "reg:squarederror",
			"--rounds", "10", "--depth", "2",
			"--out-model", filepath.Join(dir, "m-"+tag+".leaves.json"),
			"--runs", runs, "--tag", tag,
		}); err != nil {
			t.Fatalf("train %s: %v", tag, err)
		}
	}
	b, err := os.ReadFile(runs)
	if err != nil {
		t.Fatalf("runs ledger: %v", err)
	}
	lines := strings.Split(strings.TrimSpace(string(b)), "\n")
	if len(lines) != 2 {
		t.Fatalf("expected 2 ledger lines, got %d", len(lines))
	}
	var rec runRecord
	if err := json.Unmarshal([]byte(lines[0]), &rec); err != nil {
		t.Fatalf("ledger record: %v", err)
	}
	if rec.Tag != "baseline" || rec.Metric != "rmse" {
		t.Fatalf("ledger record wrong: %+v", rec)
	}
}
