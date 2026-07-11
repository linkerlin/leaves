package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

// TestSniffFeatureDimContract 锁定 WP-01：n_features == Matrix.NumCol()，
// 且与 feature_names 长度一致（标签/qid 不计入特征维）。
func TestSniffFeatureDimContract(t *testing.T) {
	dir := t.TempDir()

	// 3 特征 + label-last → n_features=3（旧 bug 会报 2）
	regPath := filepath.Join(dir, "reg.csv")
	writeCSV(t, regPath, "x0,x1,x2,label", []string{
		"0,0,0,0",
		"0.3,0.2,0.1,0.73",
		"0.6,0.4,0.2,0.91",
	})
	out := filepath.Join(dir, "reg.sniff.json")
	if err := cmdSniff([]string{"--data", regPath, "--metrics", out}); err != nil {
		t.Fatalf("sniff reg: %v", err)
	}
	var doc map[string]any
	mustJSON(t, out, &doc)
	if got := intFrom(doc["n_features"]); got != 3 {
		t.Fatalf("n_features=%d want 3 (labels are separate from NumCol)", got)
	}
	if got := intFrom(doc["n_cols"]); got != 3 {
		t.Fatalf("n_cols=%d want 3 (aligned with Matrix feature dim)", got)
	}
	names, _ := doc["feature_names"].([]any)
	if len(names) != 3 {
		t.Fatalf("feature_names len=%d want 3: %v", len(names), names)
	}
	if doc["has_label"] != true {
		t.Fatalf("has_label want true")
	}
	if doc["has_qid"] != false {
		t.Fatalf("has_qid want false for CSV")
	}

	// 排序：qid label f0 f1 → n_features=2
	rankPath := filepath.Join(dir, "rank.tsv")
	var lines []string
	for qid := 0; qid < 3; qid++ {
		for _, rel := range []int{3, 1, 0} {
			lines = append(lines, strconv.Itoa(qid)+"\t"+strconv.Itoa(rel)+"\t0.5\t0.6")
		}
	}
	if err := os.WriteFile(rankPath, []byte(strings.Join(lines, "\n")+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	rout := filepath.Join(dir, "rank.sniff.json")
	if err := cmdSniff([]string{"--data", rankPath, "--metrics", rout}); err != nil {
		t.Fatalf("sniff rank: %v", err)
	}
	var rdoc map[string]any
	mustJSON(t, rout, &rdoc)
	if got := intFrom(rdoc["n_features"]); got != 2 {
		t.Fatalf("rank n_features=%d want 2", got)
	}
	if rdoc["has_qid"] != true {
		t.Fatalf("rank has_qid want true")
	}
	if rdoc["suggested_objective"] != "rank:ndcg" {
		t.Fatalf("rank objective: %v", rdoc["suggested_objective"])
	}
}

// TestCLIMultiTarget 锁定 LIB-21 CLI：--num-target 2 末两列标签，模型 n_groups=2，predict 出 margins。
func TestCLIMultiTarget(t *testing.T) {
	dir := t.TempDir()
	var rows []string
	for i := 0; i < 40; i++ {
		x0 := float64(i%8) * 0.1
		x1 := float64(i%5) * 0.2
		y0 := x0 + 0.1*x1
		y1 := 2*x1 - 0.05*x0
		rows = append(rows, f2(x0)+","+f2(x1)+","+f2(y0)+","+f2(y1))
	}
	p := filepath.Join(dir, "mt.csv")
	writeCSV(t, p, "x0,x1,y0,y1", rows)
	// 仅特征的预测输入
	var featRows []string
	for i := 0; i < 5; i++ {
		featRows = append(featRows, f2(float64(i)*0.1)+","+f2(float64(i)*0.05))
	}
	featPath := filepath.Join(dir, "feat.csv")
	writeCSV(t, featPath, "x0,x1", featRows)

	modelPath := filepath.Join(dir, "mt.leaves.json")
	metricsPath := filepath.Join(dir, "mt.json")
	if err := cmdTrain([]string{
		"--data", p, "--objective", "reg:squarederror",
		"--num-target", "2",
		"--rounds", "20", "--depth", "3", "--lr", "0.25",
		"--out-model", modelPath, "--metrics", metricsPath,
	}); err != nil {
		t.Fatalf("train multi-target: %v", err)
	}
	var doc metricsDoc
	mustJSON(t, metricsPath, &doc)
	if doc.Params == nil || doc.Params.NumTarget != 2 {
		t.Fatalf("params.num_target want 2: %+v", doc.Params)
	}
	insp := filepath.Join(dir, "insp.json")
	if err := cmdInspect([]string{"--model", modelPath, "--metrics", insp}); err != nil {
		t.Fatalf("inspect: %v", err)
	}
	var idoc map[string]any
	mustJSON(t, insp, &idoc)
	ng := intFrom(idoc["n_output_groups"])
	if ng <= 0 {
		ng = intFrom(idoc["n_raw_output_groups"])
	}
	if ng != 2 {
		t.Fatalf("inspect output groups=%d want 2; doc=%v", ng, idoc)
	}

	predOut := filepath.Join(dir, "pred.jsonl")
	if err := cmdPredict([]string{
		"--model", modelPath, "--data", featPath, "--out", predOut,
		"--objective", "reg:squarederror",
	}); err != nil {
		t.Fatalf("predict: %v", err)
	}
	b, err := os.ReadFile(predOut)
	if err != nil {
		t.Fatal(err)
	}
	// 首行应含 margins/predictions，且无 class（非 multiclass）
	line := strings.Split(string(b), "\n")[0]
	if !strings.Contains(line, "margins") && !strings.Contains(line, "predictions") {
		t.Fatalf("predict jsonl missing margins: %s", line)
	}
	if strings.Contains(line, `"class"`) {
		t.Fatalf("multi-target predict should not emit class: %s", line)
	}

	// eval：带标签的多目标 holdout
	evalPath := filepath.Join(dir, "eval.json")
	if err := cmdEval([]string{
		"--model", modelPath, "--data", p,
		"--objective", "reg:squarederror", "--eval-metric", "rmse",
		"--num-target", "2", "--metrics", evalPath,
	}); err != nil {
		t.Fatalf("eval multi-target: %v", err)
	}
	var edoc metricsDoc
	mustJSON(t, evalPath, &edoc)
	if edoc.Value < 0 || edoc.Value > 5 {
		t.Fatalf("eval rmse out of range: %g", edoc.Value)
	}
}

// TestTrainAccelInMetrics 锁定 POST-13：单次 train 后 metrics 含 train_accel（训练加速，非推理后端）。
func TestTrainAccelInMetrics(t *testing.T) {
	dir := t.TempDir()
	var rows []string
	for i := 0; i < 20; i++ {
		fi := float64(i)
		rows = append(rows, f2(fi)+","+f2(fi*0.5)+","+f2(fi))
	}
	p := filepath.Join(dir, "t.csv")
	writeCSV(t, p, "x0,x1,label", rows)
	mout := filepath.Join(dir, "m.json")
	if err := cmdTrain([]string{
		"--data", p, "--objective", "reg:squarederror",
		"--rounds", "5", "--depth", "2",
		"--metrics", mout,
	}); err != nil {
		t.Fatalf("train: %v", err)
	}
	var doc metricsDoc
	mustJSON(t, mout, &doc)
	if doc.TrainAccel == "" {
		t.Fatalf("train_accel empty; want effective accel mode after Fit")
	}
	switch doc.TrainAccel {
	case "cpu", "born_cpu", "webgpu":
		// treebuilder.ResolveEffectiveAccelMode 规范名
	default:
		t.Fatalf("train_accel unexpected %q (want cpu|born_cpu|webgpu)", doc.TrainAccel)
	}
}

// TestMetricsSchemaVersion 锁定 WP-18：metrics / sniff 含 schema_version=1。
func TestMetricsSchemaVersion(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "t.csv")
	writeCSV(t, p, "x0,x1,label", []string{"0,0,0", "1,0.5,2", "2,1,4", "3,1.5,6"})
	mout := filepath.Join(dir, "m.json")
	if err := cmdTrain([]string{
		"--data", p, "--objective", "reg:squarederror",
		"--rounds", "3", "--depth", "2", "--metrics", mout,
	}); err != nil {
		t.Fatalf("train: %v", err)
	}
	var doc metricsDoc
	mustJSON(t, mout, &doc)
	if doc.SchemaVersion != metricsSchemaVersion {
		t.Fatalf("metrics schema_version=%d want %d", doc.SchemaVersion, metricsSchemaVersion)
	}
	sout := filepath.Join(dir, "s.json")
	if err := cmdSniff([]string{"--data", p, "--metrics", sout}); err != nil {
		t.Fatalf("sniff: %v", err)
	}
	var sdoc map[string]any
	mustJSON(t, sout, &sdoc)
	if intFrom(sdoc["schema_version"]) != metricsSchemaVersion {
		t.Fatalf("sniff schema_version=%v want %d", sdoc["schema_version"], metricsSchemaVersion)
	}
}

// TestParamsLedgerComplete 锁定 WP-02：metrics.params / runs 含全部 CLI 旋钮。
func TestParamsLedgerComplete(t *testing.T) {
	dir := t.TempDir()
	trainPath := filepath.Join(dir, "t.csv")
	var rows []string
	for i := 0; i < 24; i++ {
		fi := float64(i)
		rows = append(rows, f2(fi)+","+f2(fi*0.5)+","+f2(fi*1.5)+","+f2(fi*0.2))
	}
	writeCSV(t, trainPath, "x0,x1,x2,label", rows)

	metricsPath := filepath.Join(dir, "m.json")
	runsPath := filepath.Join(dir, "runs.jsonl")
	if err := cmdTrain([]string{
		"--data", trainPath, "--objective", "reg:squarederror",
		"--eval-metric", "rmse",
		"--rounds", "8", "--depth", "3", "--max-leaves", "0",
		"--lr", "0.15", "--lambda", "2.5",
		"--min-child-weight", "3", "--gamma", "0.5",
		"--max-bin", "64", "--subsample", "0.9", "--colsample", "0.8",
		"--tree-method", "hist", "--seed", "7",
		"--metrics", metricsPath,
		"--runs", runsPath, "--tag", "full_params",
	}); err != nil {
		t.Fatalf("train: %v", err)
	}

	var doc metricsDoc
	mustJSON(t, metricsPath, &doc)
	if doc.NFeatures != 3 {
		t.Fatalf("metrics n_features=%d want 3", doc.NFeatures)
	}
	p := doc.Params
	if p == nil {
		t.Fatal("params missing")
	}
	checks := []struct {
		name string
		ok   bool
	}{
		{"rounds", p.Rounds == 8},
		{"depth", p.Depth == 3},
		{"lr", p.LR == 0.15},
		{"lambda", p.Lambda == 2.5},
		{"min_child_weight", p.MinChildWeight == 3},
		{"gamma", p.Gamma == 0.5},
		{"max_bin", p.MaxBin == 64},
		{"subsample", p.Subsample == 0.9},
		{"colsample", p.Colsample == 0.8},
		{"tree_method", p.TreeMethod == "hist"},
		{"seed", p.Seed == 7},
		{"eval_metric", p.EvalMetric == "rmse"},
	}
	for _, c := range checks {
		if !c.ok {
			t.Errorf("params.%s wrong: %+v", c.name, p)
		}
	}

	// runs.jsonl 同步完备
	b, err := os.ReadFile(runsPath)
	if err != nil {
		t.Fatal(err)
	}
	var rec runRecord
	if err := json.Unmarshal(bytesTrimLine(b), &rec); err != nil {
		t.Fatalf("runs: %v", err)
	}
	if rec.Tag != "full_params" || rec.Params == nil {
		t.Fatalf("run record: %+v", rec)
	}
	if rec.Params.MinChildWeight != 3 || rec.Params.Subsample != 0.9 || rec.Params.Colsample != 0.8 {
		t.Fatalf("runs params incomplete: %+v", rec.Params)
	}
}

// TestParamsMulticlassAndRank 锁定 num_class / ndcg_k 写入账本。
func TestParamsMulticlassAndRank(t *testing.T) {
	dir := t.TempDir()

	// multi
	mp := filepath.Join(dir, "multi.csv")
	var rows []string
	for i := 0; i < 40; i++ {
		c := float64(i % 3)
		rows = append(rows, f2(c)+","+f2(c*0.5)+","+f2(c))
	}
	writeCSV(t, mp, "f0,f1,label", rows)
	mout := filepath.Join(dir, "multi.json")
	if err := cmdTrain([]string{
		"--data", mp, "--objective", "multi:softmax", "--num-class", "3",
		"--rounds", "5", "--depth", "2", "--metrics", mout,
	}); err != nil {
		t.Fatalf("multi train: %v", err)
	}
	var md metricsDoc
	mustJSON(t, mout, &md)
	if md.Params == nil || md.Params.NumClass != 3 {
		t.Fatalf("num_class not in params: %+v", md.Params)
	}

	// rank
	rp := filepath.Join(dir, "rank.tsv")
	var lines []string
	for qid := 0; qid < 6; qid++ {
		for _, rel := range []int{2, 1, 0} {
			lines = append(lines, strconv.Itoa(qid)+"\t"+strconv.Itoa(rel)+"\t"+strconv.Itoa(rel)+"\t0.1")
		}
	}
	if err := os.WriteFile(rp, []byte(strings.Join(lines, "\n")+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	rout := filepath.Join(dir, "rank.json")
	if err := cmdTrain([]string{
		"--data", rp, "--objective", "rank:ndcg", "--ndcg-k", "5",
		"--rounds", "5", "--depth", "2", "--metrics", rout,
	}); err != nil {
		t.Fatalf("rank train: %v", err)
	}
	var rd metricsDoc
	mustJSON(t, rout, &rd)
	if rd.Params == nil || rd.Params.NDCGK != 5 {
		t.Fatalf("ndcg_k not in params: %+v", rd.Params)
	}
}

func mustJSON(t *testing.T, path string, v any) {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(b, v); err != nil {
		t.Fatalf("json %s: %v\n%s", path, err, b)
	}
}

func intFrom(v any) int {
	switch x := v.(type) {
	case float64:
		return int(x)
	case int:
		return x
	case json.Number:
		i, _ := x.Int64()
		return int(i)
	default:
		return -1
	}
}

func bytesTrimLine(b []byte) []byte {
	s := strings.TrimSpace(string(b))
	// 多行时取第一行
	if i := strings.IndexByte(s, '\n'); i >= 0 {
		s = s[:i]
	}
	return []byte(s)
}

// TestSaveBestDefault 锁定 WP-03：--early-stop 默认 --save-best，
// 磁盘模型 model_round == best_round，且 stopped_round >= best_round。
func TestSaveBestDefault(t *testing.T) {
	dir := t.TempDir()
	// 可拟合训练 + 噪声验证集 → 易触发早停且 best < final
	var trainRows []string
	for i := 0; i < 40; i++ {
		x := float64(i) * 0.1
		y := 2*x + 0.01*float64(i%3)
		trainRows = append(trainRows, f2(x)+","+f2(x*0.5)+","+f2(y))
	}
	trainPath := filepath.Join(dir, "tr.csv")
	writeCSV(t, trainPath, "x0,x1,label", trainRows)

	var valRows []string
	for i := 0; i < 20; i++ {
		x := float64(i+50) * 0.1
		// 验证标签故意偏噪声，促使 val 恶化
		y := float64((i * 7) % 11)
		valRows = append(valRows, f2(x)+","+f2(x*0.3)+","+f2(y))
	}
	valPath := filepath.Join(dir, "va.csv")
	writeCSV(t, valPath, "x0,x1,label", valRows)

	modelPath := filepath.Join(dir, "m.leaves.json")
	metricsPath := filepath.Join(dir, "m.json")
	if err := cmdTrain([]string{
		"--data", trainPath, "--objective", "reg:squarederror",
		"--val", valPath, "--early-stop", "5",
		"--rounds", "80", "--depth", "4", "--lr", "0.3",
		"--out-model", modelPath, "--metrics", metricsPath,
		// 默认 save-best=true
	}); err != nil {
		t.Fatalf("train: %v", err)
	}

	var doc metricsDoc
	mustJSON(t, metricsPath, &doc)
	if doc.BestRound <= 0 {
		t.Fatalf("expected best_round>0, got %+v", doc)
	}
	if doc.StoppedRound < doc.BestRound {
		t.Fatalf("stopped_round %d < best_round %d", doc.StoppedRound, doc.BestRound)
	}
	if doc.ModelRound != doc.BestRound {
		t.Fatalf("model_round=%d want best_round=%d (save-best default)", doc.ModelRound, doc.BestRound)
	}

	// inspect：n_trees 应等于 model_round（单输出每轮 1 树）
	insp := filepath.Join(dir, "insp.json")
	if err := cmdInspect([]string{"--model", modelPath, "--metrics", insp}); err != nil {
		t.Fatalf("inspect: %v", err)
	}
	var idoc map[string]any
	mustJSON(t, insp, &idoc)
	nTrees := intFrom(idoc["n_trees"])
	if nTrees != doc.ModelRound {
		t.Fatalf("inspect n_trees=%d want model_round=%d", nTrees, doc.ModelRound)
	}
}

// TestOutFinalModel 锁定 POST-12：--out-final 在 save-best 截断前保存 final-round。
func TestOutFinalModel(t *testing.T) {
	dir := t.TempDir()
	var trainRows []string
	for i := 0; i < 40; i++ {
		x := float64(i) * 0.1
		y := 2*x + 0.01*float64(i%3)
		trainRows = append(trainRows, f2(x)+","+f2(x*0.5)+","+f2(y))
	}
	trainPath := filepath.Join(dir, "tr.csv")
	writeCSV(t, trainPath, "x0,x1,label", trainRows)
	var valRows []string
	for i := 0; i < 20; i++ {
		x := float64(i+50) * 0.1
		y := float64((i * 7) % 11)
		valRows = append(valRows, f2(x)+","+f2(x*0.3)+","+f2(y))
	}
	valPath := filepath.Join(dir, "va.csv")
	writeCSV(t, valPath, "x0,x1,label", valRows)

	bestPath := filepath.Join(dir, "best.leaves.json")
	finalPath := filepath.Join(dir, "final.leaves.json")
	metricsPath := filepath.Join(dir, "m.json")
	if err := cmdTrain([]string{
		"--data", trainPath, "--objective", "reg:squarederror",
		"--val", valPath, "--early-stop", "5",
		"--rounds", "80", "--depth", "4", "--lr", "0.3",
		"--out-model", bestPath, "--out-final", finalPath,
		"--metrics", metricsPath,
	}); err != nil {
		t.Fatalf("train: %v", err)
	}
	var doc metricsDoc
	mustJSON(t, metricsPath, &doc)
	if doc.BestRound <= 0 || doc.StoppedRound < doc.BestRound {
		t.Fatalf("expected early-stop shape: %+v", doc)
	}
	if doc.ModelRound != doc.BestRound {
		t.Fatalf("out-model should be best: model_round=%d best=%d", doc.ModelRound, doc.BestRound)
	}
	if doc.FinalModel != finalPath {
		t.Fatalf("final_model=%q want %q", doc.FinalModel, finalPath)
	}
	if doc.FinalRound != doc.StoppedRound {
		t.Fatalf("final_round=%d want stopped_round=%d", doc.FinalRound, doc.StoppedRound)
	}

	// inspect 两文件树数
	for _, tc := range []struct {
		path string
		want int
	}{
		{bestPath, doc.ModelRound},
		{finalPath, doc.FinalRound},
	} {
		insp := filepath.Join(dir, filepath.Base(tc.path)+".insp.json")
		if err := cmdInspect([]string{"--model", tc.path, "--metrics", insp}); err != nil {
			t.Fatalf("inspect %s: %v", tc.path, err)
		}
		var idoc map[string]any
		mustJSON(t, insp, &idoc)
		if got := intFrom(idoc["n_trees"]); got != tc.want {
			t.Fatalf("%s n_trees=%d want %d", tc.path, got, tc.want)
		}
	}
	if doc.FinalRound <= doc.ModelRound {
		// 允许相等（最后一轮即最优）；若严格更大则双产物才有差异
		t.Logf("note: final_round=%d model_round=%d (equal ok if last improved)", doc.FinalRound, doc.ModelRound)
	}
}

// TestErrorFormatJSON 锁定 WP-10：--error-format=json 输出可解析错误对象。
func TestErrorFormatJSON(t *testing.T) {
	// 直接测 classify + writeError 契约（不启子进程）。
	errorFormat = "json"
	defer func() { errorFormat = "text" }()

	ae := classifyError(errUsage("--data 必需"))
	if ae.Code != "usage" || ae.Retryable {
		t.Fatalf("usage classify: %+v", ae)
	}
	ae = classifyError(errAgent("objective_mismatch", "need num-class", "use sniff", false))
	if ae.Code != "objective_mismatch" {
		t.Fatalf("agent code: %s", ae.Code)
	}
	ae = classifyError(fmt.Errorf("load train: open x.csv: no such file or directory"))
	if ae.Code != "data_load" {
		t.Fatalf("data_load classify got %s", ae.Code)
	}
}

// TestErrorCodesHighFrequency 锁定 POST-10：高频错误码可分类且 exit_code 语义正确。
func TestErrorCodesHighFrequency(t *testing.T) {
	cases := []struct {
		err      error
		wantCode string
		wantExit int
	}{
		{errUsage("--objective 必需"), "usage", 1},
		{errAgent("objective_mismatch", "need num-class", "use sniff", false), "objective_mismatch", 1},
		{errAgent("cv_conflict", "cv vs val", "drop cv or val", false), "cv_conflict", 1},
		{errAgent("missing_value", "empty cell", "use --na-policy skip-row", false), "missing_value", 1},
		{errAgent("non_numeric", "parse float", "encode categories", false), "non_numeric", 1},
		{fmt.Errorf("open data: no such file or directory"), "data_load", 1},
		{fmt.Errorf("strconv.ParseFloat: invalid syntax"), "non_numeric", 1},
		{fmt.Errorf("boosting failed: hessian singular"), "internal", 2},
	}
	for _, tc := range cases {
		ae := classifyError(tc.err)
		if ae.Code != tc.wantCode {
			t.Errorf("classify(%v) code=%q want %q", tc.err, ae.Code, tc.wantCode)
		}
		// writeError 副作用写 stderr；用同等 exit 映射核对
		exit := 2
		switch ae.Code {
		case "usage", "data_load", "non_numeric", "missing_value", "objective_mismatch", "model_load", "cv_conflict":
			exit = 1
		}
		if exit != tc.wantExit {
			t.Errorf("code %s exit=%d want %d", ae.Code, exit, tc.wantExit)
		}
	}
}

// TestCVConflictStrictFlags 锁定 POST-02：--strict-flags 时 cv 与 val 冲突为 cv_conflict。
func TestCVConflictStrictFlags(t *testing.T) {
	dir := t.TempDir()
	var rows []string
	for i := 0; i < 24; i++ {
		fi := float64(i)
		rows = append(rows, f2(fi)+","+f2(fi*0.5)+","+f2(fi))
	}
	trainPath := filepath.Join(dir, "t.csv")
	valPath := filepath.Join(dir, "v.csv")
	writeCSV(t, trainPath, "x0,x1,label", rows)
	writeCSV(t, valPath, "x0,x1,label", rows[:8])

	// 默认：警告但成功（兼容）
	if err := cmdTrain([]string{
		"--data", trainPath, "--objective", "reg:squarederror",
		"--cv", "3", "--val", valPath, "--early-stop", "5",
		"--rounds", "6", "--depth", "2",
		"--metrics", filepath.Join(dir, "compat.json"),
	}); err != nil {
		t.Fatalf("default warn path should succeed: %v", err)
	}

	// --strict-flags：失败且 code=cv_conflict
	err := cmdTrain([]string{
		"--data", trainPath, "--objective", "reg:squarederror",
		"--cv", "3", "--val", valPath,
		"--strict-flags",
		"--rounds", "6", "--depth", "2",
		"--metrics", filepath.Join(dir, "strict.json"),
	})
	if err == nil {
		t.Fatal("strict-flags expected cv_conflict error")
	}
	ae := classifyError(err)
	if ae.Code != "cv_conflict" {
		t.Fatalf("strict error code=%q want cv_conflict; err=%v", ae.Code, err)
	}
}

// TestPublishReproduce 锁定 WP-11：manifest 含 reproduce / schema_version。
// 兼测 WP-19：--emit-repro-script both 写出 reproduce.ps1 / reproduce.sh。
func TestPublishReproduce(t *testing.T) {
	dir := t.TempDir()
	var rows []string
	for i := 0; i < 20; i++ {
		fi := float64(i)
		rows = append(rows, f2(fi)+","+f2(fi*2)+","+f2(fi))
	}
	trainPath := filepath.Join(dir, "t.csv")
	writeCSV(t, trainPath, "x0,x1,label", rows)
	modelPath := filepath.Join(dir, "m.leaves.json")
	metricsPath := filepath.Join(dir, "m.json")
	if err := cmdTrain([]string{
		"--data", trainPath, "--objective", "reg:squarederror",
		"--rounds", "8", "--depth", "2", "--lr", "0.2", "--lambda", "1.5",
		"--out-model", modelPath, "--metrics", metricsPath,
	}); err != nil {
		t.Fatalf("train: %v", err)
	}
	rel := filepath.Join(dir, "rel")
	if err := cmdPublish([]string{
		"--model", modelPath, "--out-dir", rel, "--version", "1.2.3",
		"--emit-repro-script", "both",
		"--metrics", metricsPath, "--data", trainPath,
	}); err != nil {
		t.Fatalf("publish: %v", err)
	}
	var man map[string]any
	mustJSON(t, filepath.Join(rel, "manifest.json"), &man)
	if man["schema_version"] == nil {
		t.Fatalf("missing schema_version: %v", man)
	}
	repro, _ := man["reproduce"].(string)
	if repro == "" || !strings.Contains(repro, "leaves train") || !strings.Contains(repro, "--lr") {
		t.Fatalf("reproduce missing/incomplete: %q", repro)
	}
	if !strings.Contains(repro, "0.2") {
		t.Fatalf("reproduce should include lr 0.2: %q", repro)
	}
	// WP-19 scripts
	for _, name := range []string{"reproduce.ps1", "reproduce.sh"} {
		b, err := os.ReadFile(filepath.Join(rel, name))
		if err != nil {
			t.Fatalf("missing %s: %v", name, err)
		}
		if !strings.Contains(string(b), "leaves train") || !strings.Contains(string(b), "0.2") {
			t.Fatalf("%s incomplete: %s", name, b)
		}
	}
}

// TestPublishPrintRepro 锁定 POST-11：--print-repro 写出完整 train 命令到 stdout。
func TestPublishPrintRepro(t *testing.T) {
	dir := t.TempDir()
	var rows []string
	for i := 0; i < 16; i++ {
		fi := float64(i)
		rows = append(rows, f2(fi)+","+f2(fi)+","+f2(fi))
	}
	trainPath := filepath.Join(dir, "t.csv")
	writeCSV(t, trainPath, "x0,x1,label", rows)
	modelPath := filepath.Join(dir, "m.leaves.json")
	metricsPath := filepath.Join(dir, "m.json")
	if err := cmdTrain([]string{
		"--data", trainPath, "--objective", "reg:squarederror",
		"--rounds", "6", "--depth", "2", "--lr", "0.15",
		"--out-model", modelPath, "--metrics", metricsPath,
	}); err != nil {
		t.Fatalf("train: %v", err)
	}

	old := os.Stdout
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	os.Stdout = w
	pubErr := cmdPublish([]string{
		"--model", modelPath, "--out-dir", filepath.Join(dir, "out"),
		"--metrics", metricsPath, "--data", trainPath,
		"--print-repro",
	})
	_ = w.Close()
	os.Stdout = old
	var buf [4096]byte
	n, _ := r.Read(buf[:])
	_ = r.Close()
	if pubErr != nil {
		t.Fatalf("publish: %v", pubErr)
	}
	out := string(buf[:n])
	if !strings.Contains(out, "leaves train") || !strings.Contains(out, "0.15") {
		t.Fatalf("print-repro incomplete: %q", out)
	}
}

// TestNAPolicySkipRowCLI 锁定 WP-20：--na-policy skip-row 丢弃含缺失行后可训练。
func TestNAPolicySkipRowCLI(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "miss.csv")
	// 中间两行含缺失 / nan
	content := "x0,x1,label\n0,0,0\n1,,1\n2,1,nan\n3,1.5,3\n4,2,4\n"
	if err := os.WriteFile(p, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
	// 默认 error 应失败
	if err := cmdTrain([]string{
		"--data", p, "--objective", "reg:squarederror",
		"--rounds", "3", "--depth", "2",
		"--metrics", filepath.Join(dir, "fail.json"),
	}); err == nil {
		t.Fatal("expected error on missing with default na-policy")
	}
	// skip-row 应成功，且 n_rows=3
	mout := filepath.Join(dir, "ok.json")
	if err := cmdTrain([]string{
		"--data", p, "--objective", "reg:squarederror",
		"--na-policy", "skip-row",
		"--rounds", "3", "--depth", "2",
		"--metrics", mout,
	}); err != nil {
		t.Fatalf("skip-row train: %v", err)
	}
	var doc metricsDoc
	mustJSON(t, mout, &doc)
	if doc.NRows != 3 {
		t.Fatalf("n_rows=%d want 3 after skip", doc.NRows)
	}
}

// TestSniffDataQuality 锁定 WP-12：sniff 含 data_quality 块。
func TestSniffDataQuality(t *testing.T) {
	dir := t.TempDir()
	// 常数特征 x1
	p := filepath.Join(dir, "q.csv")
	writeCSV(t, p, "x0,x1,label", []string{
		"0.1,1,0", "0.2,1,1", "0.3,1,0", "0.4,1,1", "0.5,1,0",
	})
	out := filepath.Join(dir, "s.json")
	if err := cmdSniff([]string{"--data", p, "--metrics", out}); err != nil {
		t.Fatalf("sniff: %v", err)
	}
	var doc map[string]any
	mustJSON(t, out, &doc)
	dq, ok := doc["data_quality"].(map[string]any)
	if !ok {
		t.Fatalf("missing data_quality: %v", doc)
	}
	if dq["numeric"] != true {
		t.Fatalf("numeric: %v", dq["numeric"])
	}
	cf, _ := dq["constant_features"].([]any)
	found := false
	for _, c := range cf {
		if c == "x1" {
			found = true
		}
	}
	if !found {
		t.Fatalf("expected constant feature x1, got %v", cf)
	}
}

// TestAgenticOptimizeLoopSmoke 锁定 WP-07：sniff → 两轮 train+runs → 选优 → eval → publish。
func TestAgenticOptimizeLoopSmoke(t *testing.T) {
	dir := t.TempDir()
	var rows []string
	for i := 0; i < 40; i++ {
		x0, x1 := float64(i)*0.1, float64(i%5)*0.2
		y := 2*x0 - 1.5*x1 + 0.1
		rows = append(rows, f2(x0)+","+f2(x1)+","+f2(y))
	}
	trainPath := filepath.Join(dir, "train.csv")
	writeCSV(t, trainPath, "x0,x1,label", rows)

	// holdout
	var hrows []string
	for i := 40; i < 50; i++ {
		x0, x1 := float64(i)*0.1, float64(i%5)*0.2
		y := 2*x0 - 1.5*x1 + 0.1
		hrows = append(hrows, f2(x0)+","+f2(x1)+","+f2(y))
	}
	holdPath := filepath.Join(dir, "hold.csv")
	writeCSV(t, holdPath, "x0,x1,label", hrows)

	sniffOut := filepath.Join(dir, "sniff.json")
	if err := cmdSniff([]string{"--data", trainPath, "--metrics", sniffOut}); err != nil {
		t.Fatalf("sniff: %v", err)
	}
	var sdoc map[string]any
	mustJSON(t, sniffOut, &sdoc)
	if sdoc["suggested_objective"] != "reg:squarederror" {
		t.Fatalf("sniff obj: %v", sdoc["suggested_objective"])
	}
	if intFrom(sdoc["n_features"]) != 2 {
		t.Fatalf("sniff n_features=%v want 2", sdoc["n_features"])
	}

	runs := filepath.Join(dir, "runs.jsonl")
	m1 := filepath.Join(dir, "m1.leaves.json")
	met1 := filepath.Join(dir, "m1.json")
	if err := cmdTrain([]string{
		"--data", trainPath, "--objective", "reg:squarederror",
		"--rounds", "15", "--depth", "2", "--lr", "0.3",
		"--out-model", m1, "--metrics", met1,
		"--runs", runs, "--tag", "baseline",
	}); err != nil {
		t.Fatalf("train baseline: %v", err)
	}
	m2 := filepath.Join(dir, "m2.leaves.json")
	met2 := filepath.Join(dir, "m2.json")
	if err := cmdTrain([]string{
		"--data", trainPath, "--objective", "reg:squarederror",
		"--rounds", "30", "--depth", "4", "--lr", "0.1",
		"--lambda", "2", "--min-child-weight", "2",
		"--out-model", m2, "--metrics", met2,
		"--runs", runs, "--tag", "tune1",
	}); err != nil {
		t.Fatalf("train tune1: %v", err)
	}

	b, err := os.ReadFile(runs)
	if err != nil {
		t.Fatal(err)
	}
	lines := strings.Split(strings.TrimSpace(string(b)), "\n")
	if len(lines) != 2 {
		t.Fatalf("runs lines=%d want 2", len(lines))
	}
	var best runRecord
	best.Value = 1e300
	var tune1 runRecord
	for _, line := range lines {
		var rec runRecord
		if err := json.Unmarshal([]byte(line), &rec); err != nil {
			t.Fatal(err)
		}
		if rec.Tag == "tune1" {
			tune1 = rec
		}
		if !rec.Maximize && rec.Value < best.Value {
			best = rec
		}
	}
	if best.Tag == "" {
		t.Fatal("no best run")
	}
	if tune1.Params == nil || tune1.Params.MinChildWeight != 2 || tune1.Params.Lambda != 2 {
		t.Fatalf("tune1 params incomplete: %+v", tune1.Params)
	}
	if best.Params == nil || best.Params.Subsample == 0 {
		t.Fatalf("best params incomplete: %+v", best.Params)
	}

	// 用最优模型路径做 eval + publish
	bestModel := best.Model
	if bestModel == "" {
		bestModel = m2
	}
	evalOut := filepath.Join(dir, "hold.json")
	if err := cmdEval([]string{
		"--model", bestModel, "--data", holdPath,
		"--eval-metric", "rmse", "--metrics", evalOut,
	}); err != nil {
		t.Fatalf("eval: %v", err)
	}
	rel := filepath.Join(dir, "release")
	if err := cmdPublish([]string{
		"--model", bestModel, "--out-dir", rel, "--version", "0.1.0",
		"--export-xgb", "--quantize", "--data", trainPath,
		"--metrics", met2,
	}); err != nil {
		t.Fatalf("publish: %v", err)
	}
	for _, name := range []string{"model.leaves.json", "model.xgb.json", "model.quant.json", "manifest.json"} {
		if _, err := os.Stat(filepath.Join(rel, name)); err != nil {
			t.Fatalf("missing publish artifact %s: %v", name, err)
		}
	}
}

// TestSaveBestFalse 验证 --save-best=false 保留 final-round 树数。
func TestSaveBestFalse(t *testing.T) {
	dir := t.TempDir()
	var trainRows []string
	for i := 0; i < 40; i++ {
		x := float64(i) * 0.1
		trainRows = append(trainRows, f2(x)+","+f2(x*0.5)+","+f2(2*x))
	}
	trainPath := filepath.Join(dir, "tr.csv")
	writeCSV(t, trainPath, "x0,x1,label", trainRows)

	var valRows []string
	for i := 0; i < 20; i++ {
		x := float64(i+50) * 0.1
		valRows = append(valRows, f2(x)+","+f2(x*0.3)+","+f2(float64((i*7)%11)))
	}
	valPath := filepath.Join(dir, "va.csv")
	writeCSV(t, valPath, "x0,x1,label", valRows)

	modelPath := filepath.Join(dir, "m.leaves.json")
	metricsPath := filepath.Join(dir, "m.json")
	if err := cmdTrain([]string{
		"--data", trainPath, "--objective", "reg:squarederror",
		"--val", valPath, "--early-stop", "5",
		"--rounds", "80", "--depth", "4", "--lr", "0.3",
		"--save-best=false",
		"--out-model", modelPath, "--metrics", metricsPath,
	}); err != nil {
		t.Fatalf("train: %v", err)
	}
	var doc metricsDoc
	mustJSON(t, metricsPath, &doc)
	if doc.BestRound <= 0 {
		t.Fatalf("expected early stop best_round>0: %+v", doc)
	}
	if doc.ModelRound != doc.StoppedRound {
		t.Fatalf("save-best=false: model_round=%d want stopped_round=%d", doc.ModelRound, doc.StoppedRound)
	}
	if doc.StoppedRound <= doc.BestRound {
		// 早停应在 best 之后又跑了 patience 轮（或至少不小于）
		// patience=5 时通常 stopped > best；若恰好最后一轮改进则 equal 也可
		t.Logf("note: stopped=%d best=%d (equal ok if last round improved)", doc.StoppedRound, doc.BestRound)
	}
}

// TestFromRunReproduce 锁定 WP-17：--from-run 按 tag/最优加载 params，CLI 覆盖优先。
func TestFromRunReproduce(t *testing.T) {
	dir := t.TempDir()
	var rows []string
	for i := 0; i < 30; i++ {
		x := float64(i) * 0.1
		rows = append(rows, f2(x)+","+f2(x*0.5)+","+f2(2*x+0.1))
	}
	trainPath := filepath.Join(dir, "t.csv")
	writeCSV(t, trainPath, "x0,x1,label", rows)

	runsPath := filepath.Join(dir, "runs.jsonl")
	// 源 run：depth=3 lr=0.15 lambda=2 seed=7
	if err := cmdTrain([]string{
		"--data", trainPath, "--objective", "reg:squarederror",
		"--rounds", "6", "--depth", "3", "--lr", "0.15", "--lambda", "2",
		"--seed", "7",
		"--metrics", filepath.Join(dir, "src.json"),
		"--runs", runsPath, "--tag", "src_depth3",
	}); err != nil {
		t.Fatalf("source train: %v", err)
	}

	// 1) 按 tag 复现：不传 depth/lr/objective，应继承账本
	reproMetrics := filepath.Join(dir, "repro.json")
	if err := cmdTrain([]string{
		"--data", trainPath,
		"--from-run", runsPath, "--tag", "src_depth3",
		"--rounds", "6",
		"--metrics", reproMetrics,
	}); err != nil {
		t.Fatalf("from-run by tag: %v", err)
	}
	var doc metricsDoc
	mustJSON(t, reproMetrics, &doc)
	if doc.Objective != "reg:squarederror" {
		t.Fatalf("objective from ledger: %q", doc.Objective)
	}
	if doc.Params == nil || doc.Params.Depth != 3 || doc.Params.LR != 0.15 || doc.Params.Lambda != 2 {
		t.Fatalf("params not loaded from run: %+v", doc.Params)
	}
	if doc.Params.Seed != 7 {
		t.Fatalf("seed not loaded: %d", doc.Params.Seed)
	}

	// 2) CLI 覆盖：显式 --depth 5 应压过账本 depth=3
	overMetrics := filepath.Join(dir, "over.json")
	if err := cmdTrain([]string{
		"--data", trainPath,
		"--from-run", runsPath, "--tag", "src_depth3",
		"--depth", "5", "--rounds", "4",
		"--metrics", overMetrics,
	}); err != nil {
		t.Fatalf("from-run override: %v", err)
	}
	var odoc metricsDoc
	mustJSON(t, overMetrics, &odoc)
	if odoc.Params == nil || odoc.Params.Depth != 5 || odoc.Params.LR != 0.15 {
		t.Fatalf("CLI override failed: %+v", odoc.Params)
	}

	// 3) 无 tag：按 maximize 自动选最优（用可控账本，不依赖训练偶然序）
	ledger := filepath.Join(dir, "pick.jsonl")
	// value 越小越好；best 行 depth=4，worse 行 depth=1
	ledgerBody := strings.Join([]string{
		`{"tag":"worse","objective":"reg:squarederror","metric":"rmse","value":0.9,"maximize":false,"params":{"rounds":4,"depth":1,"lr":0.3,"lambda":1,"min_child_weight":1,"gamma":0,"max_bin":256,"subsample":1,"colsample":1,"tree_method":"auto","seed":42,"eval_metric":"rmse"}}`,
		`{"tag":"best","objective":"reg:squarederror","metric":"rmse","value":0.1,"maximize":false,"params":{"rounds":4,"depth":4,"lr":0.2,"lambda":1,"min_child_weight":1,"gamma":0,"max_bin":256,"subsample":1,"colsample":1,"tree_method":"auto","seed":42,"eval_metric":"rmse"}}`,
		`{"tag":"mid","objective":"reg:squarederror","metric":"rmse","value":0.5,"maximize":false,"params":{"rounds":4,"depth":2,"lr":0.3,"lambda":1,"min_child_weight":1,"gamma":0,"max_bin":256,"subsample":1,"colsample":1,"tree_method":"auto","seed":42,"eval_metric":"rmse"}}`,
	}, "\n") + "\n"
	if err := os.WriteFile(ledger, []byte(ledgerBody), 0o644); err != nil {
		t.Fatal(err)
	}
	bestMetrics := filepath.Join(dir, "best.json")
	if err := cmdTrain([]string{
		"--data", trainPath,
		"--from-run", ledger, // 无 --tag → 应选 value=0.1 的 best
		"--rounds", "4",
		"--metrics", bestMetrics,
	}); err != nil {
		t.Fatalf("from-run best: %v", err)
	}
	var bdoc metricsDoc
	mustJSON(t, bestMetrics, &bdoc)
	if bdoc.Params == nil || bdoc.Params.Depth != 4 || bdoc.Params.LR != 0.2 {
		t.Fatalf("best pick want depth=4 lr=0.2: %+v", bdoc.Params)
	}
}

// TestOutModelMetricsSamePath 防止 metrics 覆盖模型文件。
func TestOutModelMetricsSamePath(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "same.json")
	writeCSV(t, filepath.Join(dir, "t.csv"), "x0,label", []string{"0,0", "1,1", "2,2"})
	err := cmdTrain([]string{
		"--data", filepath.Join(dir, "t.csv"), "--objective", "reg:squarederror",
		"--rounds", "2", "--depth", "2",
		"--out-model", p, "--metrics", p,
	})
	if err == nil {
		t.Fatal("expected error when out-model == metrics")
	}
	if !strings.Contains(err.Error(), "同一路径") {
		t.Fatalf("unexpected error: %v", err)
	}
}
