package main

import (
	"encoding/csv"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	leavesio "github.com/linkerlin/leaves/v2/io"
)

// TestAN1_MissingLabelAndColumnControl 锁定 WP-22 / DoD AN1：
//  1. 含自定义标签列名 y_reg 的 CSV 在未指定 --label-col 时，train/eval 必须报 missing_label（退出码 1），绝不静默训练全零模型；
//  2. sniff 未指定 --label-col 时报告 has_label=false 且 warnings 含 missing_label；
//  3. 指定 --label-col y_reg --drop-cols user_id 后，sniff/train/eval 全链路通过且自动剔除字符串 ID 列。
func TestAN1_MissingLabelAndColumnControl(t *testing.T) {
	dir := t.TempDir()
	csvPath := filepath.Join(dir, "custom.csv")
	var rows []string
	for i := 0; i < 30; i++ {
		x0 := float64(i) * 0.2
		x1 := float64(i%5) * 0.5
		y := 3.0*x0 - x1 + 1.0
		rows = append(rows, "user_"+strconv.Itoa(i)+","+f2(x0)+","+f2(x1)+","+f2(y))
	}
	writeCSV(t, csvPath, "user_id,x0,x1,y_reg", rows)

	// 1) 仅数值列但表头为 x0,x1,y_reg（无预设标签名）：train 未传 --label-col 必须拦截并报 missing_label
	numCSV := filepath.Join(dir, "num_unrecognized_label.csv")
	var numRows []string
	for i := 0; i < 20; i++ {
		x0 := float64(i) * 0.1
		numRows = append(numRows, f2(x0)+","+f2(x0*2)+","+f2(x0*3+1))
	}
	writeCSV(t, numCSV, "x0,x1,y_reg", numRows)

	err := cmdTrain([]string{
		"--data", numCSV, "--objective", "reg:squarederror",
		"--rounds", "5", "--out-model", filepath.Join(dir, "bad.leaves.json"),
	})
	if err == nil {
		t.Fatal("expected missing_label error when training on CSV without recognized label column")
	}
	var ae *agentError
	if !errors.As(err, &ae) || ae.Code != "missing_label" {
		t.Fatalf("expected agentError code=missing_label, got %v", err)
	}
	if !strings.Contains(ae.Hint, "y_reg") || !strings.Contains(ae.Hint, "--label-col") {
		t.Fatalf("expected hint to list header columns including y_reg, got %q", ae.Hint)
	}
	if code := writeError(err); code != 1 {
		t.Fatalf("expected exit code 1 for missing_label, got %d", code)
	}

	// 2) sniff 在未识别标签列时必须报告 has_label=false 与 warning=missing_label
	sniffNoLabel := filepath.Join(dir, "sniff_nolabel.json")
	if err := cmdSniff([]string{"--data", numCSV, "--metrics", sniffNoLabel}); err != nil {
		t.Fatalf("sniff without label: %v", err)
	}
	var sdoc map[string]any
	mustJSON(t, sniffNoLabel, &sdoc)
	if sdoc["has_label"] != false {
		t.Fatalf("expected has_label=false, got %v", sdoc["has_label"])
	}

	// 3) 传入 --label-col y_reg --drop-cols user_id：sniff → train → eval 全链路畅通
	sniffOK := filepath.Join(dir, "sniff_ok.json")
	if err := cmdSniff([]string{
		"--data", csvPath, "--label-col", "y_reg", "--drop-cols", "user_id", "--metrics", sniffOK,
	}); err != nil {
		t.Fatalf("sniff with label-col and drop-cols: %v", err)
	}
	mustJSON(t, sniffOK, &sdoc)
	if sdoc["has_label"] != true || intFrom(sdoc["n_features"]) != 2 {
		t.Fatalf("sniff expected has_label=true n_features=2, got %+v", sdoc)
	}

	modelPath := filepath.Join(dir, "m.leaves.json")
	metricsPath := filepath.Join(dir, "m.json")
	if err := cmdTrain([]string{
		"--data", csvPath, "--objective", "reg:squarederror",
		"--label-col", "y_reg", "--drop-cols", "user_id",
		"--rounds", "15", "--depth", "3",
		"--out-model", modelPath, "--metrics", metricsPath,
	}); err != nil {
		t.Fatalf("train with label-col & drop-cols: %v", err)
	}
	var mdoc metricsDoc
	mustJSON(t, metricsPath, &mdoc)
	if mdoc.NFeatures != 2 || mdoc.Params.LabelCol != "y_reg" || mdoc.Params.DropCols != "user_id" {
		t.Fatalf("unexpected train metricsDoc: %+v", mdoc)
	}

	evalOut := filepath.Join(dir, "eval.json")
	if err := cmdEval([]string{
		"--model", modelPath, "--data", csvPath,
		"--label-col", "y_reg", "--drop-cols", "user_id",
		"--metrics", evalOut,
	}); err != nil {
		t.Fatalf("eval with label-col & drop-cols: %v", err)
	}
}

// TestAN2_ZeroLeakageSplit 锁定 WP-23 / DoD AN2：
// 验证 stratified / group / time 三种零泄漏切分策略，且 group 策略下无任何 qid 跨集出现。
func TestAN2_ZeroLeakageSplit(t *testing.T) {
	dir := t.TempDir()
	rawCSV := filepath.Join(dir, "raw.csv")
	var rows []string
	for qid := 0; qid < 12; qid++ {
		for j := 0; j < 4; j++ {
			ts := qid*10 + j
			lbl := j % 2
			rows = append(rows, strconv.Itoa(qid)+","+strconv.Itoa(ts)+","+f2(float64(qid)*0.1)+","+strconv.Itoa(lbl))
		}
	}
	writeCSV(t, rawCSV, "qid,ts,x0,label", rows)

	// 1) stratified 切分
	stratDir := filepath.Join(dir, "split_strat")
	if err := cmdSplit([]string{
		"--data", rawCSV, "--out-dir", stratDir,
		"--strategy", "stratified", "--ratios", "0.7,0.15,0.15",
		"--metrics", filepath.Join(dir, "strat_rep.json"),
	}); err != nil {
		t.Fatalf("split stratified: %v", err)
	}
	var srep map[string]any
	mustJSON(t, filepath.Join(stratDir, "split.json"), &srep)
	tot := intFrom(srep["train_rows"]) + intFrom(srep["val_rows"]) + intFrom(srep["test_rows"])
	if tot != 48 {
		t.Fatalf("stratified rows sum=%d want 48", tot)
	}

	// 2) group 切分：任何 qid 绝不跨 train/val/test 出现
	grpDir := filepath.Join(dir, "split_group")
	if err := cmdSplit([]string{
		"--data", rawCSV, "--out-dir", grpDir,
		"--strategy", "group", "--group-col", "qid", "--ratios", "0.6,0.2,0.2",
		"--metrics", filepath.Join(dir, "grp_rep.json"),
	}); err != nil {
		t.Fatalf("split group: %v", err)
	}
	var grep map[string]any
	mustJSON(t, filepath.Join(grpDir, "split.json"), &grep)
	gtot := intFrom(grep["train_rows"]) + intFrom(grep["val_rows"]) + intFrom(grep["test_rows"])
	if gtot != 48 {
		t.Fatalf("group rows sum=%d want 48", gtot)
	}
	seenSet := map[string]string{}
	for _, sub := range []string{"train", "val", "test"} {
		p := filepath.Join(grpDir, sub+".csv")
		f, err := os.Open(p)
		if err != nil {
			t.Fatal(err)
		}
		recs, err := csv.NewReader(f).ReadAll()
		f.Close()
		if err != nil {
			t.Fatal(err)
		}
		for _, r := range recs[1:] {
			q := r[0]
			if prev, ok := seenSet[q]; ok && prev != sub {
				t.Fatalf("group leakage: qid %s appeared in both %s and %s", q, prev, sub)
			}
			seenSet[q] = sub
		}
	}

	// 3) time 切分：时间戳严格升序且不交叠
	timeDir := filepath.Join(dir, "split_time")
	if err := cmdSplit([]string{
		"--data", rawCSV, "--out-dir", timeDir,
		"--strategy", "time", "--time-col", "ts", "--ratios", "0.5,0.25,0.25",
		"--metrics", filepath.Join(dir, "time_rep.json"),
	}); err != nil {
		t.Fatalf("split time: %v", err)
	}
	maxTS := func(p string) float64 {
		f, _ := os.Open(p)
		defer f.Close()
		recs, _ := csv.NewReader(f).ReadAll()
		mx := -1e9
		for _, r := range recs[1:] {
			v, _ := strconv.ParseFloat(r[1], 64)
			if v > mx {
				mx = v
			}
		}
		return mx
	}
	minTS := func(p string) float64 {
		f, _ := os.Open(p)
		defer f.Close()
		recs, _ := csv.NewReader(f).ReadAll()
		mn := 1e9
		for _, r := range recs[1:] {
			v, _ := strconv.ParseFloat(r[1], 64)
			if v < mn {
				mn = v
			}
		}
		return mn
	}
	if maxTS(filepath.Join(timeDir, "train.csv")) > minTS(filepath.Join(timeDir, "val.csv")) ||
		maxTS(filepath.Join(timeDir, "val.csv")) > minTS(filepath.Join(timeDir, "test.csv")) {
		t.Fatal("time split violated chronological ordering")
	}
}

// TestAN3_FromRunBestRoundFinalization 锁定 WP-24 / DoD AN3：
// 早停 run 写入 runs.jsonl 后，使用 --from-run 全量定稿（不传 --val 与 --rounds）时，
// 最终模型树数自动锁定为 best_round 而非初始最大轮数上限（80）。
func TestAN3_FromRunBestRoundFinalization(t *testing.T) {
	dir := t.TempDir()
	var trRows, vaRows []string
	for i := 0; i < 40; i++ {
		x := float64(i) * 0.1
		trRows = append(trRows, f2(x)+","+f2(x*0.5)+","+f2(2*x))
	}
	for i := 0; i < 20; i++ {
		x := float64(i+50) * 0.1
		vaRows = append(vaRows, f2(x)+","+f2(x*0.3)+","+f2(float64((i*7)%11)))
	}
	trPath := filepath.Join(dir, "tr.csv")
	vaPath := filepath.Join(dir, "va.csv")
	writeCSV(t, trPath, "x0,x1,label", trRows)
	writeCSV(t, vaPath, "x0,x1,label", vaRows)

	runsPath := filepath.Join(dir, "runs.jsonl")
	m1 := filepath.Join(dir, "m1.leaves.json")
	met1 := filepath.Join(dir, "m1.json")
	if err := cmdTrain([]string{
		"--data", trPath, "--val", vaPath, "--objective", "reg:squarederror",
		"--early-stop", "4", "--rounds", "80", "--depth", "4", "--lr", "0.3",
		"--out-model", m1, "--metrics", met1,
		"--runs", runsPath, "--tag", "es_parent",
	}); err != nil {
		t.Fatalf("train early-stop parent: %v", err)
	}
	var doc1 metricsDoc
	mustJSON(t, met1, &doc1)
	if doc1.BestRound <= 0 || doc1.BestRound >= 80 {
		t.Fatalf("expected early stop best_round in (0,80), got %d", doc1.BestRound)
	}

	// 全量定稿：--from-run runs.jsonl，不传 --val 也不传 --rounds
	finalModel := filepath.Join(dir, "final.leaves.json")
	finalMet := filepath.Join(dir, "final.json")
	if err := cmdTrain([]string{
		"--data", trPath, "--from-run", runsPath, "--tag", "es_parent",
		"--out-model", finalModel, "--metrics", finalMet,
	}); err != nil {
		t.Fatalf("train final from-run: %v", err)
	}
	var docFinal metricsDoc
	mustJSON(t, finalMet, &docFinal)
	if docFinal.NTrees != doc1.BestRound {
		t.Fatalf("AN3 failed: final model n_trees=%d, want best_round=%d (not max rounds 80)", docFinal.NTrees, doc1.BestRound)
	}
}

// TestAN4_HonestGroupAndStratifiedCV 锁定 WP-24 / DoD AN4：
// 验证排序数据上的 GroupKFold + 折内早停（--cv 3 --early-stop 3），以及二分类 StratifiedKFold。
func TestAN4_HonestGroupAndStratifiedCV(t *testing.T) {
	dir := t.TempDir()
	rankPath := filepath.Join(dir, "rank.tsv")
	var lines []string
	for qid := 0; qid < 12; qid++ {
		for _, rel := range []int{3, 2, 1, 0} {
			lines = append(lines, strconv.Itoa(qid)+"\t"+strconv.Itoa(rel)+"\t"+
				f2(float64(rel)*0.8)+"\t"+f2(float64(qid%3)*0.2))
		}
	}
	if err := os.WriteFile(rankPath, []byte(strings.Join(lines, "\n")+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	rankModel := filepath.Join(dir, "rank_cv.leaves.json")
	rankMet := filepath.Join(dir, "rank_cv.json")
	runsPath := filepath.Join(dir, "runs.jsonl")
	if err := cmdTrain([]string{
		"--data", rankPath, "--objective", "rank:ndcg", "--ndcg-k", "4",
		"--cv", "3", "--early-stop", "3", "--rounds", "25", "--depth", "3",
		"--out-model", rankModel, "--metrics", rankMet,
		"--runs", runsPath, "--tag", "rank_cv_es",
	}); err != nil {
		t.Fatalf("ranking GroupKFold CV with early-stop failed: %v", err)
	}
	var rdoc metricsDoc
	mustJSON(t, rankMet, &rdoc)
	if len(rdoc.FoldMetrics) != 3 || len(rdoc.CVBestRounds) != 3 || rdoc.CVBestRoundMean <= 0 {
		t.Fatalf("expected 3 fold_metrics and cv_best_rounds, got %+v", rdoc)
	}
	if rdoc.NTrees != rdoc.BestRound {
		t.Fatalf("expected CV out-model n_trees=%d to match mean best_round=%d", rdoc.NTrees, rdoc.BestRound)
	}
}

// TestAN5_RunsAndDiagnoseNativePrimitives 锁定 WP-25 / DoD AN5：
// 验证 leaves runs <status|pareto|select-parent|check-dup> 与 leaves diagnose。
func TestAN5_RunsAndDiagnoseNativePrimitives(t *testing.T) {
	dir := t.TempDir()
	runsPath := filepath.Join(dir, "runs.jsonl")
	ledgerLines := []string{
		`{"tag":"base","objective":"reg:squarederror","metric":"rmse","value":0.50,"maximize":false,"fold_metrics":[0.52,0.48,0.50],"n_trees":50,"elapsed_ms":20,"params":{"rounds":50,"depth":6,"lr":0.3,"lambda":1,"min_child_weight":1,"gamma":0,"max_bin":256,"subsample":1,"colsample":1,"tree_method":"auto","seed":42}}`,
		`{"tag":"m1_deep","objective":"reg:squarederror","metric":"rmse","value":0.42,"maximize":false,"fold_metrics":[0.40,0.46,0.40],"n_trees":50,"elapsed_ms":30,"params":{"rounds":50,"depth":8,"lr":0.3,"lambda":1,"min_child_weight":1,"gamma":0,"max_bin":256,"subsample":1,"colsample":1,"tree_method":"auto","seed":42}}`,
		`{"tag":"m2_compact","objective":"reg:squarederror","metric":"rmse","value":0.43,"maximize":false,"fold_metrics":[0.45,0.39,0.45],"n_trees":20,"elapsed_ms":10,"params":{"rounds":20,"depth":4,"lr":0.3,"lambda":2,"min_child_weight":1,"gamma":0,"max_bin":256,"subsample":1,"colsample":1,"tree_method":"auto","seed":42}}`,
		`{"tag":"m3_stale1","objective":"reg:squarederror","metric":"rmse","value":0.42,"maximize":false,"fold_metrics":[0.41,0.45,0.40],"n_trees":50,"elapsed_ms":30,"params":{"rounds":50,"depth":8,"lr":0.25,"lambda":1,"min_child_weight":1,"gamma":0,"max_bin":256,"subsample":1,"colsample":1,"tree_method":"auto","seed":42}}`,
		`{"tag":"m4_stale2","objective":"reg:squarederror","metric":"rmse","value":0.42,"maximize":false,"fold_metrics":[0.41,0.45,0.40],"n_trees":50,"elapsed_ms":30,"params":{"rounds":50,"depth":8,"lr":0.2,"lambda":1,"min_child_weight":1,"gamma":0,"max_bin":256,"subsample":1,"colsample":1,"tree_method":"auto","seed":42}}`,
	}
	if err := os.WriteFile(runsPath, []byte(strings.Join(ledgerLines, "\n")+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	// 1) runs status
	statusOut := filepath.Join(dir, "status.json")
	if err := cmdRuns([]string{"status", "--runs", runsPath, "--metrics", statusOut}); err != nil {
		t.Fatalf("runs status: %v", err)
	}
	var st map[string]any
	mustJSON(t, statusOut, &st)
	if intFrom(st["total_runs"]) != 5 || st["best_tag"] != "m1_deep" || st["converged"] != true {
		t.Fatalf("unexpected runs status: %+v", st)
	}

	// 2) runs pareto（m1_deep 在折 0/2 最优，m2_compact 在折 1 最优且树数更少，均在前沿上）
	paretoOut := filepath.Join(dir, "pareto.json")
	if err := cmdRuns([]string{"pareto", "--runs", runsPath, "--metrics", paretoOut}); err != nil {
		t.Fatalf("runs pareto: %v", err)
	}
	var par map[string]any
	mustJSON(t, paretoOut, &par)
	ftags, _ := par["fold_pareto_tags"].([]any)
	if len(ftags) < 2 {
		t.Fatalf("expected >=2 fold pareto candidates, got %v", ftags)
	}

	// 3) runs select-parent
	selOut := filepath.Join(dir, "select.json")
	if err := cmdRuns([]string{"select-parent", "--runs", runsPath, "--seed", "7", "--metrics", selOut}); err != nil {
		t.Fatalf("runs select-parent: %v", err)
	}
	var sel map[string]any
	mustJSON(t, selOut, &sel)
	if sel["selected_tag"] == nil || sel["selected_tag"] == "" {
		t.Fatalf("select-parent missing selected_tag: %+v", sel)
	}

	// 4) runs check-dup
	dupOut := filepath.Join(dir, "dup.json")
	if err := cmdRuns([]string{"check-dup", "--runs", runsPath, "--depth", "8", "--lr", "0.3", "--metrics", dupOut}); err != nil {
		t.Fatalf("runs check-dup: %v", err)
	}
	var dup map[string]any
	mustJSON(t, dupOut, &dup)
	if dup["duplicate"] != true || dup["matched_tag"] != "m1_deep" {
		t.Fatalf("expected duplicate match m1_deep, got %+v", dup)
	}

	// 5) diagnose 过拟合学习曲线
	roundsLog := filepath.Join(dir, "rounds.jsonl")
	rLines := []string{
		`{"round":0,"lr":0.3,"train":0.80,"val":0.82}`,
		`{"round":1,"lr":0.3,"train":0.50,"val":0.55}`,
		`{"round":2,"lr":0.3,"train":0.30,"val":0.40}`,
		`{"round":3,"lr":0.3,"train":0.18,"val":0.45}`,
		`{"round":4,"lr":0.3,"train":0.10,"val":0.52}`,
		`{"round":5,"lr":0.3,"train":0.05,"val":0.60}`,
	}
	if err := os.WriteFile(roundsLog, []byte(strings.Join(rLines, "\n")+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	diagOut := filepath.Join(dir, "diag.json")
	if err := cmdDiagnose([]string{"--rounds-log", roundsLog, "--out", diagOut}); err != nil {
		t.Fatalf("diagnose: %v", err)
	}
	var diag map[string]any
	mustJSON(t, diagOut, &diag)
	if diag["diagnosis"] != "overfitting" || intFrom(diag["best_round"]) != 3 {
		t.Fatalf("expected overfitting with best_round=3, got %+v", diag)
	}
}

// TestAN6_ExternalModelCLIAndMulticlassSHAP 锁定 WP-26 / DoD AN6：
//  1. 外部 XGBoost JSON 模型直接执行 inspect / explain / eval / publish 零报错；
//  2. 多分类模型（multi:softprob）直接执行 explain --type shap 输出多类 SHAP 矩阵。
func TestAN6_ExternalModelCLIAndMulticlassSHAP(t *testing.T) {
	dir := t.TempDir()
	trPath := filepath.Join(dir, "tr.csv")
	var rows []string
	for i := 0; i < 30; i++ {
		x := float64(i) * 0.1
		rows = append(rows, f2(x)+","+f2(x*0.5)+","+f2(2*x+0.5))
	}
	writeCSV(t, trPath, "x0,x1,label", rows)

	leavesModel := filepath.Join(dir, "m.leaves.json")
	if err := cmdTrain([]string{
		"--data", trPath, "--objective", "reg:squarederror",
		"--rounds", "10", "--depth", "3", "--out-model", leavesModel,
	}); err != nil {
		t.Fatalf("train: %v", err)
	}
	ir, obj, err := leavesio.ParseLeavesJSONFile(leavesModel)
	if err != nil {
		t.Fatal(err)
	}
	xgbModel := filepath.Join(dir, "external.xgb.json")
	if err := leavesio.ExportXGBoostJSONFile(xgbModel, ir, obj); err != nil {
		t.Fatalf("export xgb: %v", err)
	}

	// 对外部 XGBoost JSON 模型直接运行 inspect / explain / eval / publish
	inspOut := filepath.Join(dir, "insp.json")
	if err := cmdInspect([]string{"--model", xgbModel, "--metrics", inspOut}); err != nil {
		t.Fatalf("inspect external xgb: %v", err)
	}
	impOut := filepath.Join(dir, "imp.json")
	if err := cmdExplain([]string{"--model", xgbModel, "--type", "importance", "--metrics", impOut}); err != nil {
		t.Fatalf("explain importance external xgb: %v", err)
	}
	shapOut := filepath.Join(dir, "shap.json")
	if err := cmdExplain([]string{"--model", xgbModel, "--type", "shap", "--data", trPath, "--max-rows", "5", "--metrics", shapOut}); err != nil {
		t.Fatalf("explain shap external xgb: %v", err)
	}
	evalOut := filepath.Join(dir, "eval.json")
	if err := cmdEval([]string{"--model", xgbModel, "--data", trPath, "--metrics", evalOut}); err != nil {
		t.Fatalf("eval external xgb: %v", err)
	}
	pubDir := filepath.Join(dir, "pub_xgb")
	if err := cmdPublish([]string{"--model", xgbModel, "--out-dir", pubDir, "--quantize", "--data", trPath}); err != nil {
		t.Fatalf("publish external xgb: %v", err)
	}

	// 多分类 Tree SHAP
	mcPath := filepath.Join(dir, "mc.csv")
	var mcRows []string
	for i := 0; i < 36; i++ {
		c := i % 3
		mcRows = append(mcRows, f2(float64(c)*1.2)+","+f2(float64(i)*0.1)+","+strconv.Itoa(c))
	}
	writeCSV(t, mcPath, "f0,f1,label", mcRows)
	mcModel := filepath.Join(dir, "mc.leaves.json")
	if err := cmdTrain([]string{
		"--data", mcPath, "--objective", "multi:softprob", "--num-class", "3",
		"--rounds", "8", "--depth", "3", "--out-model", mcModel,
	}); err != nil {
		t.Fatalf("train multiclass: %v", err)
	}
	mcShapOut := filepath.Join(dir, "mc_shap.json")
	if err := cmdExplain([]string{
		"--model", mcModel, "--type", "shap", "--data", mcPath, "--max-rows", "4", "--metrics", mcShapOut,
	}); err != nil {
		t.Fatalf("multiclass explain shap: %v", err)
	}
	var mcShap []map[string]any
	mustJSON(t, mcShapOut, &mcShap)
	if len(mcShap) != 4 {
		t.Fatalf("expected 4 shap rows, got %d", len(mcShap))
	}
	classes, ok := mcShap[0]["classes"].([]any)
	if !ok || len(classes) != 3 {
		t.Fatalf("expected 3 classes in multiclass shap output, got %+v", mcShap[0])
	}
}

// TestAN7_PredictIDAndOutputModesAndTrainFlags 锁定 WP-27 / DoD AN7：
//  1. predict --id-col 透传字符串 ID 且支持 --output value|margin|leaf|contrib；
//  2. train 支持 --booster dart / --monotone / --lr-decay / --batch-rows / --checkpoint / --resume。
func TestAN7_PredictIDAndOutputModesAndTrainFlags(t *testing.T) {
	dir := t.TempDir()
	dataPath := filepath.Join(dir, "with_id.csv")
	var rows []string
	for i := 0; i < 24; i++ {
		x0 := float64(i) * 0.2
		x1 := float64(24-i) * 0.1
		y := 2*x0 - x1
		rows = append(rows, "uid_"+strconv.Itoa(100+i)+","+f2(x0)+","+f2(x1)+","+f2(y))
	}
	writeCSV(t, dataPath, "uid,x0,x1,label", rows)

	ckptPath := filepath.Join(dir, "ckpt.json")
	modelPath := filepath.Join(dir, "dart.leaves.json")
	metPath := filepath.Join(dir, "dart.json")
	if err := cmdTrain([]string{
		"--data", dataPath, "--objective", "reg:squarederror",
		"--drop-cols", "uid",
		"--booster", "dart", "--rate-drop", "0.1",
		"--monotone", "1,-1", "--lr-decay", "0.95",
		"--batch-rows", "8",
		"--checkpoint", ckptPath, "--checkpoint-every", "3",
		"--rounds", "6", "--depth", "3",
		"--out-model", modelPath, "--metrics", metPath,
	}); err != nil {
		t.Fatalf("train dart+monotone+batch+ckpt: %v", err)
	}
	if _, err := os.Stat(ckptPath); err != nil {
		t.Fatalf("checkpoint not created: %v", err)
	}

	// 从 checkpoint 续训至 10 轮
	resumedModel := filepath.Join(dir, "resumed.leaves.json")
	if err := cmdTrain([]string{
		"--data", dataPath, "--drop-cols", "uid",
		"--resume", ckptPath, "--rounds", "10", "--depth", "3",
		"--out-model", resumedModel, "--metrics", filepath.Join(dir, "resumed.json"),
	}); err != nil {
		t.Fatalf("resume train: %v", err)
	}

	// predict --id-col uid --output value|leaf|contrib
	for _, mode := range []string{"value", "leaf", "contrib"} {
		outPath := filepath.Join(dir, "pred_"+mode+".jsonl")
		if err := cmdPredict([]string{
			"--model", resumedModel, "--data", dataPath,
			"--id-col", "uid", "--output", mode, "--out", outPath,
		}); err != nil {
			t.Fatalf("predict mode=%s: %v", mode, err)
		}
		b, err := os.ReadFile(outPath)
		if err != nil {
			t.Fatal(err)
		}
		lines := strings.Split(strings.TrimSpace(string(b)), "\n")
		if len(lines) != 24 {
			t.Fatalf("mode=%s lines=%d want 24", mode, len(lines))
		}
		var first map[string]any
		if err := json.Unmarshal([]byte(lines[0]), &first); err != nil {
			t.Fatal(err)
		}
		if first["id"] != "uid_100" {
			t.Fatalf("mode=%s expected id=uid_100, got %v", mode, first["id"])
		}
		switch mode {
		case "leaf":
			leaves, ok := first["leaves"].([]any)
			if !ok || len(leaves) != 10 {
				t.Fatalf("leaf mode expected 10 tree leaves, got %v", first["leaves"])
			}
		case "contrib":
			contribs, ok := first["contributions"].([]any)
			if !ok || len(contribs) != 2 {
				t.Fatalf("contrib mode expected 2 feature contribs, got %v", first["contributions"])
			}
		}
	}
}

// TestWP29_SchemaAndDryRun 锁定 WP-29：leaves schema 与 leaves train --dry-run。
func TestWP29_SchemaAndDryRun(t *testing.T) {
	dir := t.TempDir()
	schemaOut := filepath.Join(dir, "schema.json")
	if err := cmdSchema([]string{"--metrics", schemaOut}); err != nil {
		t.Fatalf("schema: %v", err)
	}
	var sdoc map[string]any
	mustJSON(t, schemaOut, &sdoc)
	cmds, ok := sdoc["commands"].(map[string]any)
	if !ok || cmds["train"] == nil || cmds["split"] == nil || cmds["runs"] == nil || cmds["diagnose"] == nil {
		t.Fatalf("schema missing expected commands: %+v", sdoc)
	}

	trPath := filepath.Join(dir, "tr.csv")
	writeCSV(t, trPath, "x0,x1,label", []string{"0.1,0.2,1", "0.3,0.4,0", "0.5,0.6,1"})
	dryOut := filepath.Join(dir, "dry.json")
	if err := cmdTrain([]string{
		"--data", trPath, "--objective", "binary:logistic",
		"--dry-run", "--metrics", dryOut,
	}); err != nil {
		t.Fatalf("train --dry-run: %v", err)
	}
	var ddoc map[string]any
	mustJSON(t, dryOut, &ddoc)
	if ddoc["dry_run"] != true || ddoc["valid"] != true {
		t.Fatalf("unexpected dry-run output: %+v", ddoc)
	}
}

// TestAD1_FeatureNamesPersistenceAndPublishColControl 锁定 WP-30 / DoD AD1：
//  1. 训练后 leaves.json 持久化真实表头特征名，inspect 与 explain（无需 --data）直接输出真实列名；
//  2. publish --quantize --data --metrics 自动从 metrics.params 继承 --label-col 与 --drop-cols 完成量化 parity。
func TestAD1_FeatureNamesPersistenceAndPublishColControl(t *testing.T) {
	dir := t.TempDir()
	csvPath := filepath.Join(dir, "biz.csv")
	var rows []string
	for i := 0; i < 30; i++ {
		age := float64(20 + i)
		inc := float64(50 + i*2)
		price := 1.5*age + 0.5*inc
		rows = append(rows, fmt.Sprintf("user_%d,%g,%g,%g", i, age, inc, price))
	}
	writeCSV(t, csvPath, "user_id,age_years,income_k,house_price", rows)

	modelPath := filepath.Join(dir, "m.leaves.json")
	metricsPath := filepath.Join(dir, "m.metrics.json")
	if err := cmdTrain([]string{
		"--data", csvPath, "--objective", "reg:squarederror",
		"--label-col", "house_price", "--drop-cols", "user_id",
		"--rounds", "10", "--depth", "3",
		"--out-model", modelPath, "--metrics", metricsPath,
	}); err != nil {
		t.Fatalf("train: %v", err)
	}

	// 1. inspect 应包含 feature_names: ["age_years", "income_k"]
	inspOut := filepath.Join(dir, "insp.json")
	if err := cmdInspect([]string{"--model", modelPath, "--metrics", inspOut}); err != nil {
		t.Fatalf("inspect: %v", err)
	}
	var inspDoc map[string]any
	mustJSON(t, inspOut, &inspDoc)
	fnames, _ := inspDoc["feature_names"].([]any)
	if len(fnames) != 2 || fnames[0] != "age_years" || fnames[1] != "income_k" {
		t.Fatalf("expected persisted feature_names [age_years, income_k], got %v", fnames)
	}

	// 2. explain --type importance（不传 --data）应直接输出真实特征名
	impOut := filepath.Join(dir, "imp.json")
	if err := cmdExplain([]string{"--model", modelPath, "--type", "importance", "--metrics", impOut}); err != nil {
		t.Fatalf("explain importance: %v", err)
	}
	var impDoc map[string]any
	mustJSON(t, impOut, &impDoc)
	feats, _ := impDoc["features"].([]any)
	if len(feats) != 2 {
		t.Fatalf("expected 2 features in importance, got %v", feats)
	}
	f0, _ := feats[0].(map[string]any)
	if f0["name"] != "age_years" {
		t.Fatalf("expected first feature name age_years, got %v", f0["name"])
	}

	// 3. publish --quantize --data csvPath --metrics metricsPath（不显式传 --label-col/--drop-cols，应自动从 metrics 继承）
	relDir := filepath.Join(dir, "rel")
	if err := cmdPublish([]string{
		"--model", modelPath, "--out-dir", relDir,
		"--quantize", "--data", csvPath, "--metrics", metricsPath,
	}); err != nil {
		t.Fatalf("publish quantize: %v", err)
	}
	var qrep map[string]any
	mustJSON(t, filepath.Join(relDir, "quantize_report.json"), &qrep)
	parity, ok := qrep["parity"].(map[string]any)
	if !ok || parity["pass"] != true {
		t.Fatalf("expected quantize parity pass=true via inherited col control, got %+v", qrep)
	}
}

// TestAD2_NAPolicyNaNAndAutoEncodeCategories 锁定 WP-31 / DoD AD2：
//  1. 含空单元格与字符串类别特征（city, device）的 CSV 通过 --na-policy nan --auto-encode 直接训练；
//  2. category_maps 随模型持久化，eval 与 predict（含未知新类别 OOV）无需额外参数即可一致编码推理。
func TestAD2_NAPolicyNaNAndAutoEncodeCategories(t *testing.T) {
	dir := t.TempDir()
	trainCSV := filepath.Join(dir, "dirty_train.csv")
	cities := []string{"Beijing", "Shanghai", "Shenzhen"}
	devices := []string{"iOS", "Android"}
	var rows []string
	for i := 0; i < 36; i++ {
		city := cities[i%len(cities)]
		dev := devices[i%len(devices)]
		numCell := fmt.Sprintf("%g", float64(i)*0.5)
		if i%7 == 0 {
			numCell = "" // 缺失单元格，走 --na-policy nan
		}
		y := 0
		if city == "Shanghai" || i > 18 {
			y = 1
		}
		rows = append(rows, fmt.Sprintf("%s,%s,%s,%d", city, dev, numCell, y))
	}
	writeCSV(t, trainCSV, "city,device,spend,label", rows)

	modelPath := filepath.Join(dir, "cat.leaves.json")
	metricsPath := filepath.Join(dir, "cat.metrics.json")
	if err := cmdTrain([]string{
		"--data", trainCSV, "--objective", "binary:logistic",
		"--na-policy", "nan", "--auto-encode",
		"--rounds", "12", "--depth", "3",
		"--out-model", modelPath, "--metrics", metricsPath,
	}); err != nil {
		t.Fatalf("train with nan + auto-encode: %v", err)
	}

	var mdoc metricsDoc
	mustJSON(t, metricsPath, &mdoc)
	if mdoc.NRows != 36 {
		t.Fatalf("na-policy=nan should keep all 36 rows, got %d", mdoc.NRows)
	}

	// 构造含缺失值与未知新类别（"Hangzhou", "HarmonyOS"）的测试集
	testCSV := filepath.Join(dir, "dirty_test.csv")
	writeCSV(t, testCSV, "city,device,spend,label", []string{
		"Shanghai,iOS,12.5,1",
		"Beijing,Android,,0",
		"Hangzhou,HarmonyOS,8.0,1", // OOV 新类别 → 自动映射为 NaN
	})

	// eval 无需再传 --auto-encode，自动从模型读取 category_maps
	evalOut := filepath.Join(dir, "eval.json")
	if err := cmdEval([]string{
		"--model", modelPath, "--data", testCSV,
		"--na-policy", "nan", "--metrics", evalOut,
	}); err != nil {
		t.Fatalf("eval with persisted category_maps: %v", err)
	}

	predOut := filepath.Join(dir, "pred.jsonl")
	if err := cmdPredict([]string{
		"--model", modelPath, "--data", testCSV,
		"--na-policy", "nan", "--out", predOut,
	}); err != nil {
		t.Fatalf("predict with persisted category_maps: %v", err)
	}
	b, err := os.ReadFile(predOut)
	if err != nil {
		t.Fatal(err)
	}
	lines := strings.Split(strings.TrimSpace(string(b)), "\n")
	if len(lines) != 3 {
		t.Fatalf("expected 3 prediction lines, got %d", len(lines))
	}
}

// TestAD3_BinaryDiagnosticsAndThreshold 锁定 WP-32 / DoD AD3：
//  1. binary:logistic 训练与评估输出 binary_diagnostics（含 auc, best_threshold, best_f1, confusion_matrix）；
//  2. predict 支持 --threshold 并输出 predicted_label。
func TestAD3_BinaryDiagnosticsAndThreshold(t *testing.T) {
	dir := t.TempDir()
	csvPath := filepath.Join(dir, "bin.csv")
	var rows []string
	for i := 0; i < 40; i++ {
		x0 := float64(i) * 0.1
		x1 := float64(i%4) * 0.2
		y := 0
		if x0+x1 > 2.0 {
			y = 1
		}
		rows = append(rows, fmt.Sprintf("%g,%g,%d", x0, x1, y))
	}
	writeCSV(t, csvPath, "x0,x1,label", rows)

	modelPath := filepath.Join(dir, "bin.leaves.json")
	metricsPath := filepath.Join(dir, "bin.metrics.json")
	if err := cmdTrain([]string{
		"--data", csvPath, "--objective", "binary:logistic",
		"--rounds", "15", "--depth", "3",
		"--out-model", modelPath, "--metrics", metricsPath,
	}); err != nil {
		t.Fatalf("train binary: %v", err)
	}
	var mdoc metricsDoc
	mustJSON(t, metricsPath, &mdoc)
	if mdoc.BinaryDiagnostics == nil {
		t.Fatalf("expected binary_diagnostics in train metrics")
	}
	if mdoc.BinaryDiagnostics.AUC < 0.8 || mdoc.BinaryDiagnostics.BestF1 < 0.7 {
		t.Fatalf("unexpected binary diagnostics: %+v", mdoc.BinaryDiagnostics)
	}
	if mdoc.BinaryDiagnostics.ConfusionMatrix["tp"]+mdoc.BinaryDiagnostics.ConfusionMatrix["tn"] <= 0 {
		t.Fatalf("empty confusion matrix: %+v", mdoc.BinaryDiagnostics.ConfusionMatrix)
	}

	predOut := filepath.Join(dir, "pred.jsonl")
	if err := cmdPredict([]string{
		"--model", modelPath, "--data", csvPath,
		"--threshold", fmt.Sprintf("%g", mdoc.BinaryDiagnostics.BestThreshold),
		"--out", predOut,
	}); err != nil {
		t.Fatalf("predict with threshold: %v", err)
	}
	b, _ := os.ReadFile(predOut)
	firstLine := strings.Split(strings.TrimSpace(string(b)), "\n")[0]
	var rec map[string]any
	if err := json.Unmarshal([]byte(firstLine), &rec); err != nil {
		t.Fatal(err)
	}
	if _, ok := rec["predicted_label"]; !ok {
		t.Fatalf("expected predicted_label in binary prediction JSONL, got %+v", rec)
	}
}

// TestAD4_CompareModelsWithBootstrap 锁定 WP-33 / DoD AD4：
// leaves compare 在同一数据集上横向对比多个模型，输出 winner_model、delta_pct 与配对 Bootstrap 胜率。
func TestAD4_CompareModelsWithBootstrap(t *testing.T) {
	dir := t.TempDir()
	csvPath := filepath.Join(dir, "reg.csv")
	var rows []string
	for i := 0; i < 50; i++ {
		x0 := float64(i) * 0.1
		x1 := float64(i%7) * 0.3
		y := 3.0*x0 - 1.5*x1 + 0.5
		rows = append(rows, fmt.Sprintf("%g,%g,%g", x0, x1, y))
	}
	writeCSV(t, csvPath, "x0,x1,label", rows)

	mWeak := filepath.Join(dir, "weak.leaves.json")
	mStrong := filepath.Join(dir, "strong.leaves.json")
	if err := cmdTrain([]string{
		"--data", csvPath, "--objective", "reg:squarederror",
		"--rounds", "2", "--depth", "1", "--lr", "0.1",
		"--out-model", mWeak, "--metrics", filepath.Join(dir, "w.json"),
	}); err != nil {
		t.Fatalf("train weak: %v", err)
	}
	if err := cmdTrain([]string{
		"--data", csvPath, "--objective", "reg:squarederror",
		"--rounds", "30", "--depth", "4", "--lr", "0.2",
		"--out-model", mStrong, "--metrics", filepath.Join(dir, "s.json"),
	}); err != nil {
		t.Fatalf("train strong: %v", err)
	}

	cmpOut := filepath.Join(dir, "compare.json")
	if err := cmdCompare([]string{
		"--models", mWeak + "," + mStrong,
		"--data", csvPath,
		"--bootstrap", "100",
		"--metrics", cmpOut,
	}); err != nil {
		t.Fatalf("compare: %v", err)
	}
	var cdoc map[string]any
	mustJSON(t, cmpOut, &cdoc)
	if cdoc["winner_model"] != mStrong || intFrom(cdoc["winner_index"]) != 1 {
		t.Fatalf("expected strong model to win, got %+v", cdoc)
	}
	models, _ := cdoc["models"].([]any)
	if len(models) != 2 {
		t.Fatalf("expected 2 model entries, got %v", models)
	}
	m1, _ := models[1].(map[string]any)
	if m1["better_than_baseline"] != true || m1["significant"] != true {
		t.Fatalf("expected strong model to be significantly better than baseline: %+v", m1)
	}
}

// TestAO1_HolisticDiagnoseWithModelAndData 锁定 WP-35 / DoD AO1：
// leaves diagnose --model --rounds --metrics --data 自动检测零增益特征、常数列、ID 列并产出 suggested_drop_cols 与 suggested_cli_args。
func TestAO1_HolisticDiagnoseWithModelAndData(t *testing.T) {
	dir := t.TempDir()
	csvPath := filepath.Join(dir, "train.csv")
	var rows []string
	for i := 0; i < 36; i++ {
		x0 := float64(i%9) * 0.4
		x1 := float64(i%5) * 0.6
		constCol := 7.0
		y := 2.0*x0 - 1.2*x1 + 0.3
		rows = append(rows, fmt.Sprintf("u_%d,%g,%g,%g,0,%g", 1000+i, x0, x1, constCol, y))
	}
	writeCSV(t, csvPath, "user_id,x0,x1,const_col,noise_zero,label", rows)

	modelPath := filepath.Join(dir, "m.leaves.json")
	metricsPath := filepath.Join(dir, "m.json")
	roundsPath := filepath.Join(dir, "rounds.jsonl")
	if err := cmdTrain([]string{
		"--data", csvPath, "--objective", "reg:squarederror",
		"--drop-cols", "user_id",
		"--rounds", "10", "--depth", "3", "--lr", "0.25",
		"--emit-rounds", roundsPath,
		"--out-model", modelPath, "--metrics", metricsPath,
	}); err != nil {
		t.Fatalf("train: %v", err)
	}

	diagOut := filepath.Join(dir, "diag.json")
	// 不传 --drop-cols 给 diagnose，让 diagnose 同时从 --data 探测出 user_id 与 const_col，并从 --model 探测出 zero_gain_features
	if err := cmdDiagnose([]string{
		"--model", modelPath,
		"--rounds", roundsPath,
		"--metrics", metricsPath,
		"--data", csvPath,
		"--drop-cols", "",
		"--out", diagOut,
	}); err != nil {
		t.Fatalf("diagnose: %v", err)
	}
	var ddoc map[string]any
	mustJSON(t, diagOut, &ddoc)
	zg, _ := ddoc["zero_gain_features"].([]any)
	if len(zg) == 0 {
		t.Fatalf("expected zero_gain_features to be non-empty, got %+v", ddoc)
	}
	suggDrops, _ := ddoc["suggested_drop_cols"].([]any)
	if len(suggDrops) == 0 {
		t.Fatalf("expected suggested_drop_cols in diagnose output: %+v", ddoc)
	}
	cliArgs, _ := ddoc["suggested_cli_args"].([]any)
	hasDropFlag := false
	for _, a := range cliArgs {
		if a == "--drop-cols" {
			hasDropFlag = true
		}
	}
	if !hasDropFlag {
		t.Fatalf("expected --drop-cols in suggested_cli_args: %v", cliArgs)
	}
}

// TestAO2_RunsCrossoverAndPropose 锁定 WP-36 / DoD AO2：
//  1. leaves runs check-dup --params '{...}' 支持 JSON 超参去重检测；
//  2. leaves runs crossover 组合父代 A 结构旋钮与父代 B 正则旋钮，产出 x:A|B 谱系标签且不与账本重复；
//  3. leaves runs propose 结合诊断报告产出 p:parent+mutation 候选且不与账本重复。
func TestAO2_RunsCrossoverAndPropose(t *testing.T) {
	dir := t.TempDir()
	runsPath := filepath.Join(dir, "runs.jsonl")
	rA := runRecord{
		Tag: "cand_a", Objective: "reg:squarederror", Metric: "rmse", Value: 0.22, Maximize: false,
		FoldMetrics: []float64{0.20, 0.24}, NTrees: 40,
		Params: &paramsRecord{Rounds: 40, Depth: 4, LR: 0.15, Lambda: 1.0, MinChildWeight: 1.0, Subsample: 1.0, Colsample: 1.0},
	}
	rB := runRecord{
		Tag: "cand_b", Objective: "reg:squarederror", Metric: "rmse", Value: 0.23, Maximize: false,
		FoldMetrics: []float64{0.25, 0.21}, NTrees: 80,
		Params: &paramsRecord{Rounds: 80, Depth: 7, LR: 0.08, Lambda: 5.0, MinChildWeight: 3.0, Subsample: 0.8, Colsample: 0.8},
	}
	bA, _ := json.Marshal(rA)
	bB, _ := json.Marshal(rB)
	if err := os.WriteFile(runsPath, []byte(string(bA)+"\n"+string(bB)+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	// 1. check-dup with --params JSON
	dupOut := filepath.Join(dir, "dup.json")
	if err := cmdRuns([]string{"check-dup", "--runs", runsPath, "--params", `{"depth":4,"lr":0.15,"lambda":1.0}`, "--metrics", dupOut}); err != nil {
		t.Fatalf("runs check-dup --params: %v", err)
	}
	var dupDoc map[string]any
	mustJSON(t, dupOut, &dupDoc)
	if dupDoc["duplicate"] != true || dupDoc["matched_tag"] != "cand_a" {
		t.Fatalf("expected duplicate=true matched_tag=cand_a, got %+v", dupDoc)
	}

	// 2. crossover
	crossOut := filepath.Join(dir, "cross.json")
	if err := cmdRuns([]string{"crossover", "--runs", runsPath, "--parent-a", "cand_a", "--parent-b", "cand_b", "--metrics", crossOut}); err != nil {
		t.Fatalf("runs crossover: %v", err)
	}
	var crossDoc map[string]any
	mustJSON(t, crossOut, &crossDoc)
	if crossDoc["suggested_tag"] != "x:cand_a|cand_b" || crossDoc["duplicate_in_ledger"] != false {
		t.Fatalf("unexpected crossover output: %+v", crossDoc)
	}
	cp, _ := crossDoc["params"].(map[string]any)
	if intFrom(cp["depth"]) != 4 || cp["lambda"] != float64(5) || cp["subsample"] != 0.8 {
		t.Fatalf("expected child to inherit depth=4 from A and lambda=5, subsample=0.8 from B, got %+v", cp)
	}

	// 3. propose with diagnose
	diagPath := filepath.Join(dir, "diag.json")
	_ = writeJSON(diagPath, map[string]any{
		"diagnosis":           "overfitting",
		"suggested_drop_cols": []string{"noise_col"},
	})
	propOut := filepath.Join(dir, "propose.json")
	if err := cmdRuns([]string{"propose", "--runs", runsPath, "--parent", "cand_a", "--diagnose", diagPath, "--metrics", propOut}); err != nil {
		t.Fatalf("runs propose: %v", err)
	}
	var propDoc map[string]any
	mustJSON(t, propOut, &propDoc)
	sTag, _ := propDoc["suggested_tag"].(string)
	if !strings.HasPrefix(sTag, "p:cand_a+") || propDoc["duplicate_in_ledger"] != false {
		t.Fatalf("unexpected propose output: %+v", propDoc)
	}
	pp, _ := propDoc["params"].(map[string]any)
	if pp["drop_cols"] != "noise_col" {
		t.Fatalf("expected propose to inherit suggested_drop_cols=noise_col, got %+v", pp)
	}
}

// TestAO3_ExplainInteractionSummaryAndDump 锁定 WP-37 / DoD AO3：
// leaves explain 支持多维重要性（weight/cover/gain_pct）、shap --summary、interaction 二阶交互与 dump 树导出。
func TestAO3_ExplainInteractionSummaryAndDump(t *testing.T) {
	dir := t.TempDir()
	csvPath := filepath.Join(dir, "train.csv")
	var rows []string
	for i := 0; i < 32; i++ {
		x0 := float64(i%8) * 0.3
		x1 := float64(i%4) * 0.5
		y := 3.0*x0*x1 + x0
		rows = append(rows, fmt.Sprintf("%g,%g,0,%g", x0, x1, y))
	}
	writeCSV(t, csvPath, "x0,x1,x_unused,label", rows)

	modelPath := filepath.Join(dir, "m.leaves.json")
	if err := cmdTrain([]string{
		"--data", csvPath, "--objective", "reg:squarederror",
		"--rounds", "15", "--depth", "3", "--lr", "0.25",
		"--out-model", modelPath, "--metrics", filepath.Join(dir, "m.json"),
	}); err != nil {
		t.Fatalf("train: %v", err)
	}

	// 1. importance with weight + ranked_features + zero_gain_features
	impOut := filepath.Join(dir, "imp.json")
	if err := cmdExplain([]string{"--model", modelPath, "--type", "importance", "--importance-type", "weight", "--metrics", impOut}); err != nil {
		t.Fatalf("explain importance weight: %v", err)
	}
	var impDoc map[string]any
	mustJSON(t, impOut, &impDoc)
	if impDoc["type"] != "weight" {
		t.Fatalf("expected type=weight, got %v", impDoc["type"])
	}
	ranked, _ := impDoc["ranked_features"].([]any)
	zg, _ := impDoc["zero_gain_features"].([]any)
	if len(ranked) != 3 || len(zg) == 0 {
		t.Fatalf("expected 3 ranked_features and non-empty zero_gain_features, got %+v", impDoc)
	}

	// 2. shap --summary
	sumOut := filepath.Join(dir, "shap_sum.json")
	if err := cmdExplain([]string{"--model", modelPath, "--type", "shap", "--summary", "--data", csvPath, "--metrics", sumOut}); err != nil {
		t.Fatalf("explain shap --summary: %v", err)
	}
	var sumDoc map[string]any
	mustJSON(t, sumOut, &sumDoc)
	if sumDoc["type"] != "shap_summary" {
		t.Fatalf("expected type=shap_summary, got %v", sumDoc["type"])
	}
	sFeats, _ := sumDoc["features"].([]any)
	if len(sFeats) != 3 {
		t.Fatalf("expected 3 features in shap_summary, got %v", len(sFeats))
	}
	f0, _ := sFeats[0].(map[string]any)
	f2, _ := sFeats[2].(map[string]any)
	if f0["mean_abs_shap"].(float64) <= f2["mean_abs_shap"].(float64) {
		t.Fatalf("expected features sorted descending by mean_abs_shap: %v vs %v", f0, f2)
	}

	// 3. interaction
	interOut := filepath.Join(dir, "inter.json")
	if err := cmdExplain([]string{"--model", modelPath, "--type", "interaction", "--data", csvPath, "--metrics", interOut}); err != nil {
		t.Fatalf("explain interaction: %v", err)
	}
	var interDoc map[string]any
	mustJSON(t, interOut, &interDoc)
	topPairs, _ := interDoc["top_interactions"].([]any)
	if len(topPairs) == 0 {
		t.Fatalf("expected top_interactions in interaction output: %+v", interDoc)
	}

	// 4. dump (dot & text)
	dotOut := filepath.Join(dir, "forest.dot")
	if err := cmdExplain([]string{"--model", modelPath, "--type", "dump", "--dump-format", "dot", "--metrics", dotOut}); err != nil {
		t.Fatalf("explain dump dot: %v", err)
	}
	dotBytes, _ := os.ReadFile(dotOut)
	if !strings.Contains(string(dotBytes), "digraph forest") || !strings.Contains(string(dotBytes), "x0") {
		t.Fatalf("expected DOT output with real feature name x0, got: %s", string(dotBytes))
	}
}

// TestAO4_RegressionAndMulticlassDiagnostics 锁定 WP-38 / DoD AO4：
// 回归与多分类在 train/eval 中自动输出 regression_diagnostics 与 multiclass_diagnostics。
func TestAO4_RegressionAndMulticlassDiagnostics(t *testing.T) {
	dir := t.TempDir()
	regCSV := filepath.Join(dir, "reg.csv")
	var regRows []string
	for i := 0; i < 30; i++ {
		x0 := float64(i) * 0.2
		y := 2.0*x0 + 1.0
		regRows = append(regRows, fmt.Sprintf("%g,%g,%g", x0, x0*0.5, y))
	}
	writeCSV(t, regCSV, "x0,x1,label", regRows)

	regModel := filepath.Join(dir, "reg.leaves.json")
	regMet := filepath.Join(dir, "reg.metrics.json")
	if err := cmdTrain([]string{
		"--data", regCSV, "--objective", "reg:squarederror",
		"--rounds", "25", "--depth", "3", "--lr", "0.25",
		"--out-model", regModel, "--metrics", regMet,
	}); err != nil {
		t.Fatalf("train reg: %v", err)
	}
	var rdoc metricsDoc
	mustJSON(t, regMet, &rdoc)
	if rdoc.RegressionDiagnostics == nil || rdoc.RegressionDiagnostics.R2 < 0.8 || rdoc.RegressionDiagnostics.MAE <= 0 {
		t.Fatalf("expected valid regression_diagnostics with R2>=0.8, got %+v", rdoc.RegressionDiagnostics)
	}

	mcCSV := filepath.Join(dir, "mc.csv")
	var mcRows []string
	for i := 0; i < 36; i++ {
		cls := i % 3
		x0 := float64(cls)*3.0 + 0.1*float64(i%4)
		x1 := float64(2-cls)*2.0 + 0.1*float64(i%3)
		mcRows = append(mcRows, fmt.Sprintf("%g,%g,%d", x0, x1, cls))
	}
	writeCSV(t, mcCSV, "x0,x1,label", mcRows)

	mcModel := filepath.Join(dir, "mc.leaves.json")
	mcMet := filepath.Join(dir, "mc.metrics.json")
	if err := cmdTrain([]string{
		"--data", mcCSV, "--objective", "multi:softprob", "--num-class", "3",
		"--rounds", "15", "--depth", "3", "--lr", "0.3",
		"--out-model", mcModel, "--metrics", mcMet,
	}); err != nil {
		t.Fatalf("train mc: %v", err)
	}
	var mcdoc metricsDoc
	mustJSON(t, mcMet, &mcdoc)
	if mcdoc.MulticlassDiagnostics == nil || mcdoc.MulticlassDiagnostics.Accuracy < 0.8 || len(mcdoc.MulticlassDiagnostics.ConfusionMatrix) != 3 {
		t.Fatalf("expected valid multiclass_diagnostics with 3x3 confusion matrix, got %+v", mcdoc.MulticlassDiagnostics)
	}
}

// TestAO5_PublishExportONNXRoundTrip 锁定 WP-39 / DoD AO5：
// leaves publish --export-onnx 生成标准 model.onnx，且可被 leaves inspect/eval/predict/compare 无缝回读并对齐预测。
func TestAO5_PublishExportONNXRoundTrip(t *testing.T) {
	dir := t.TempDir()
	csvPath := filepath.Join(dir, "bin.csv")
	var rows []string
	for i := 0; i < 36; i++ {
		x0 := float64(i%9) * 0.3
		x1 := float64(i%6) * 0.4
		y := 0
		if x0+x1 > 1.8 {
			y = 1
		}
		rows = append(rows, fmt.Sprintf("%g,%g,%d", x0, x1, y))
	}
	writeCSV(t, csvPath, "x0,x1,label", rows)

	modelPath := filepath.Join(dir, "bin.leaves.json")
	metricsPath := filepath.Join(dir, "bin.metrics.json")
	if err := cmdTrain([]string{
		"--data", csvPath, "--objective", "binary:logistic",
		"--rounds", "12", "--depth", "3", "--lr", "0.25",
		"--out-model", modelPath, "--metrics", metricsPath,
	}); err != nil {
		t.Fatalf("train binary: %v", err)
	}

	relDir := filepath.Join(dir, "release")
	if err := cmdPublish([]string{
		"--model", modelPath, "--out-dir", relDir,
		"--export-xgb", "--export-onnx", "--metrics", metricsPath,
	}); err != nil {
		t.Fatalf("publish --export-onnx: %v", err)
	}

	onnxPath := filepath.Join(relDir, "model.onnx")
	if _, err := os.Stat(onnxPath); err != nil {
		t.Fatalf("expected model.onnx to exist: %v", err)
	}

	// 用 leaves compare 对照原 leaves.json 与导出的 model.onnx，验证指标一致
	cmpOut := filepath.Join(dir, "cmp_onnx.json")
	if err := cmdCompare([]string{
		"--models", modelPath + "," + onnxPath,
		"--data", csvPath,
		"--bootstrap", "50",
		"--metrics", cmpOut,
	}); err != nil {
		t.Fatalf("compare leaves.json vs model.onnx: %v", err)
	}
	var cdoc map[string]any
	mustJSON(t, cmpOut, &cdoc)
	models, _ := cdoc["models"].([]any)
	if len(models) != 2 {
		t.Fatalf("expected 2 models in compare, got %v", len(models))
	}
	v0 := models[0].(map[string]any)["value"].(float64)
	v1 := models[1].(map[string]any)["value"].(float64)
	if v0-v1 > 1e-4 || v1-v0 > 1e-4 {
		t.Fatalf("expected exported ONNX metric (%g) to match leaves.json (%g)", v1, v0)
	}
}

// TestAA1_FeatureSchemaAutoAlignAndMismatchGate 验证 WP-40 / DoD AA1：
// 模型持久化 FeatureNames 后，在 eval/predict 中面对乱序或未传 --drop-cols 的多余列 CSV 能自动投影重排，
// 面对缺失特征列 CSV 则立即报 error=feature_mismatch。
func TestAA1_FeatureSchemaAutoAlignAndMismatchGate(t *testing.T) {
	dir := t.TempDir()
	trainCSV := filepath.Join(dir, "train.csv")
	var sb strings.Builder
	sb.WriteString("user_id,feat_a,feat_b,label\n")
	for i := 0; i < 30; i++ {
		fa := float64(i)
		fb := float64(i%5) * 10.0
		y := 2.0*fa - 0.5*fb + 1.0
		sb.WriteString(fmt.Sprintf("U%03d,%.2f,%.2f,%.4f\n", i, fa, fb, y))
	}
	if err := os.WriteFile(trainCSV, []byte(sb.String()), 0o644); err != nil {
		t.Fatal(err)
	}

	modelPath := filepath.Join(dir, "m.leaves.json")
	metricsNorm := filepath.Join(dir, "m_norm.json")
	if err := cmdTrain([]string{
		"--data", trainCSV, "--objective", "reg:squarederror",
		"--drop-cols", "user_id",
		"--rounds", "15", "--depth", "4", "--lr", "0.25",
		"--out-model", modelPath, "--metrics", metricsNorm,
	}); err != nil {
		t.Fatalf("train: %v", err)
	}

	// 构造列顺序反转（feat_b 在前、feat_a 在后）且含字符串多余列 user_id 与 extra_note 的测试集
	reorderedCSV := filepath.Join(dir, "test_reordered.csv")
	var sbRev strings.Builder
	sbRev.WriteString("feat_b,user_id,extra_note,feat_a,label\n")
	for i := 0; i < 30; i++ {
		fa := float64(i)
		fb := float64(i%5) * 10.0
		y := 2.0*fa - 0.5*fb + 1.0
		sbRev.WriteString(fmt.Sprintf("%.2f,U%03d,ignored_str,%.2f,%.4f\n", fb, i, fa, y))
	}
	if err := os.WriteFile(reorderedCSV, []byte(sbRev.String()), 0o644); err != nil {
		t.Fatal(err)
	}

	// 不传 --drop-cols，直接 eval 与 predict，验证自动投影重排后指标与原训练集完全一致！
	evalRevOut := filepath.Join(dir, "eval_rev.json")
	if err := cmdEval([]string{
		"--model", modelPath,
		"--data", reorderedCSV,
		"--metrics", evalRevOut,
	}); err != nil {
		t.Fatalf("eval on reordered CSV without --drop-cols failed: %v", err)
	}
	var mTrain, mRev metricsDoc
	mustJSON(t, metricsNorm, &mTrain)
	mustJSON(t, evalRevOut, &mRev)
	if mTrain.Value-mRev.Value > 1e-6 || mRev.Value-mTrain.Value > 1e-6 {
		t.Fatalf("expected reordered eval metric (%g) to match train metric (%g)", mRev.Value, mTrain.Value)
	}

	predRevOut := filepath.Join(dir, "pred_rev.jsonl")
	if err := cmdPredict([]string{
		"--model", modelPath,
		"--data", reorderedCSV,
		"--out", predRevOut,
	}); err != nil {
		t.Fatalf("predict on reordered CSV failed: %v", err)
	}

	// 构造缺失 feat_b 的损坏测试集，验证报 feature_mismatch
	missingCSV := filepath.Join(dir, "test_missing.csv")
	if err := os.WriteFile(missingCSV, []byte("feat_a,wrong_col,label\n1.0,2.0,3.0\n2.0,4.0,5.0\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	err := cmdEval([]string{
		"--model", modelPath,
		"--data", missingCSV,
	})
	if err == nil {
		t.Fatalf("expected feature_mismatch error when required feature feat_b is missing")
	}
	ae := classifyError(err)
	if ae.Code != "feature_mismatch" {
		t.Fatalf("expected error code feature_mismatch, got %q (%v)", ae.Code, err)
	}
}

// TestAA2_SampleWeightColAndScalePosWeight 验证 WP-41 / DoD AA2：
// --weight-col 与 --scale-pos-weight 在训练、diagnose 类别不平衡建议及 --from-run 中闭环生效。
func TestAA2_SampleWeightColAndScalePosWeight(t *testing.T) {
	dir := t.TempDir()
	csvPath := filepath.Join(dir, "imbalanced.csv")
	var sb strings.Builder
	sb.WriteString("f1,f2,sample_w,label\n")
	// 32 行中仅 4 行正样本（pos_ratio = 0.125）
	for i := 0; i < 32; i++ {
		f1 := float64(i) * 0.5
		f2 := float64(i % 4)
		w := 1.0
		y := 0
		if i >= 28 {
			y = 1
			w = 2.5
		}
		sb.WriteString(fmt.Sprintf("%.2f,%.2f,%.2f,%d\n", f1, f2, w, y))
	}
	if err := os.WriteFile(csvPath, []byte(sb.String()), 0o644); err != nil {
		t.Fatal(err)
	}

	diagOut := filepath.Join(dir, "diag.json")
	if err := cmdDiagnose([]string{
		"--data", csvPath,
		"--drop-cols", "sample_w",
		"--out", diagOut,
	}); err != nil {
		t.Fatalf("diagnose imbalanced data: %v", err)
	}
	var ddoc map[string]any
	mustJSON(t, diagOut, &ddoc)
	spw, ok := ddoc["suggested_scale_pos_weight"].(float64)
	if !ok || spw < 3.0 {
		t.Fatalf("expected suggested_scale_pos_weight >= 3.0 for 12.5%% positive rate, got %v", ddoc["suggested_scale_pos_weight"])
	}

	runsPath := filepath.Join(dir, "runs.jsonl")
	m1Path := filepath.Join(dir, "m1.json")
	model1Path := filepath.Join(dir, "m1.leaves.json")
	if err := cmdTrain([]string{
		"--data", csvPath, "--objective", "binary:logistic",
		"--weight-col", "sample_w",
		"--scale-pos-weight", "7",
		"--rounds", "10", "--depth", "3",
		"--out-model", model1Path, "--metrics", m1Path,
		"--runs", runsPath, "--tag", "weighted_base",
	}); err != nil {
		t.Fatalf("train with --weight-col and --scale-pos-weight: %v", err)
	}
	var m1 metricsDoc
	mustJSON(t, m1Path, &m1)
	if m1.NFeatures != 2 {
		t.Fatalf("expected weight-col to be excluded from features (want 2 features, got %d)", m1.NFeatures)
	}
	if m1.Params == nil || m1.Params.WeightCol != "sample_w" || m1.Params.ScalePosWeight != 7 {
		t.Fatalf("expected params to record weight_col=sample_w and scale_pos_weight=7, got %+v", m1.Params)
	}

	// 验证 --from-run 自动继承 weight_col 与 scale_pos_weight
	m2Path := filepath.Join(dir, "m2.json")
	if err := cmdTrain([]string{
		"--data", csvPath,
		"--from-run", runsPath, "--tag", "weighted_base",
		"--metrics", m2Path,
	}); err != nil {
		t.Fatalf("train --from-run: %v", err)
	}
	var m2 metricsDoc
	mustJSON(t, m2Path, &m2)
	if m2.NFeatures != 2 || m2.Params == nil || m2.Params.WeightCol != "sample_w" || m2.Params.ScalePosWeight != 7 {
		t.Fatalf("expected --from-run to inherit weight_col and scale_pos_weight, got %+v", m2.Params)
	}
}

// TestAA3_RankingDiagnosticsAndAdvancedTrainFlags 验证 WP-42 / DoD AA3：
// 排序学习任务在 train 与 eval 中自动输出 ranking_diagnostics，且高级训练旋钮生效。
func TestAA3_RankingDiagnosticsAndAdvancedTrainFlags(t *testing.T) {
	dir := t.TempDir()
	rankTSV := filepath.Join(dir, "rank.tsv")
	var sb strings.Builder
	// 3 个 Query 组：qid=1 (4 行含正样本), qid=2 (4 行含正样本), qid=3 (3 行全 0 死 Query)
	rows := []string{
		"1\t3\t0.9\t0.1",
		"1\t2\t0.7\t0.2",
		"1\t0\t0.2\t0.8",
		"1\t0\t0.1\t0.9",
		"2\t2\t0.85\t0.15",
		"2\t1\t0.60\t0.40",
		"2\t0\t0.15\t0.85",
		"2\t0\t0.05\t0.95",
		"3\t0\t0.40\t0.50",
		"3\t0\t0.30\t0.60",
		"3\t0\t0.20\t0.70",
	}
	for _, r := range rows {
		sb.WriteString(r + "\n")
	}
	if err := os.WriteFile(rankTSV, []byte(sb.String()), 0o644); err != nil {
		t.Fatal(err)
	}

	rankModel := filepath.Join(dir, "rank.leaves.json")
	rankMetrics := filepath.Join(dir, "rank_metrics.json")
	if err := cmdTrain([]string{
		"--data", rankTSV, "--objective", "rank:ndcg",
		"--rounds", "10", "--depth", "3",
		"--out-model", rankModel, "--metrics", rankMetrics,
	}); err != nil {
		t.Fatalf("train rank:ndcg: %v", err)
	}
	var rdoc metricsDoc
	mustJSON(t, rankMetrics, &rdoc)
	if rdoc.RankingDiagnostics == nil {
		t.Fatalf("expected ranking_diagnostics in train metrics.json")
	}
	if rdoc.RankingDiagnostics.NQueries != 3 || rdoc.RankingDiagnostics.ZeroRelevantQueries != 1 {
		t.Fatalf("expected n_queries=3 and zero_relevant_queries=1, got %+v", rdoc.RankingDiagnostics)
	}
	if rdoc.RankingDiagnostics.NDCGAt1 <= 0 || rdoc.RankingDiagnostics.MRR <= 0 {
		t.Fatalf("expected positive ndcg_at_1 and mrr, got %+v", rdoc.RankingDiagnostics)
	}

	evalRankOut := filepath.Join(dir, "rank_eval.json")
	if err := cmdEval([]string{
		"--model", rankModel, "--data", rankTSV,
		"--metrics", evalRankOut,
	}); err != nil {
		t.Fatalf("eval rank: %v", err)
	}
	var edoc metricsDoc
	mustJSON(t, evalRankOut, &edoc)
	if edoc.RankingDiagnostics == nil || edoc.RankingDiagnostics.NQueries != 3 {
		t.Fatalf("expected ranking_diagnostics in eval metrics.json, got %+v", edoc.RankingDiagnostics)
	}

	// 验证 --tweedie-variance-power 与 --num-parallel-tree
	twCSV := filepath.Join(dir, "tweedie.csv")
	var sbTw strings.Builder
	sbTw.WriteString("x1,x2,label\n")
	for i := 1; i <= 20; i++ {
		sbTw.WriteString(fmt.Sprintf("%.2f,%.2f,%.2f\n", float64(i), float64(i%3), float64(i)*1.5))
	}
	if err := os.WriteFile(twCSV, []byte(sbTw.String()), 0o644); err != nil {
		t.Fatal(err)
	}
	twMetrics := filepath.Join(dir, "tw_metrics.json")
	if err := cmdTrain([]string{
		"--data", twCSV, "--objective", "reg:tweedie",
		"--tweedie-variance-power", "1.6",
		"--num-parallel-tree", "2",
		"--rounds", "5", "--depth", "3",
		"--metrics", twMetrics,
	}); err != nil {
		t.Fatalf("train reg:tweedie with advanced flags: %v", err)
	}
	var twDoc metricsDoc
	mustJSON(t, twMetrics, &twDoc)
	if twDoc.Params == nil || twDoc.Params.TweedieVariancePower != 1.6 || twDoc.Params.NumParallelTree != 2 {
		t.Fatalf("expected tweedie_variance_power=1.6 and num_parallel_tree=2 in params, got %+v", twDoc.Params)
	}
}

// TestAA4_AutotuneBoundedReflectiveLoop 验证 WP-43 / DoD AA4：
// leaves autotune 在含 ID 列、字符串类别列与缺失值的脏 CSV 上自动完成嗅探、基线、反射变异/交叉与定稿。
func TestAA4_AutotuneBoundedReflectiveLoop(t *testing.T) {
	dir := t.TempDir()
	dirtyCSV := filepath.Join(dir, "dirty_train.csv")
	var sb strings.Builder
	sb.WriteString("row_id,city,x1,x2,label\n")
	cities := []string{"Beijing", "Shanghai", "Shenzhen"}
	for i := 0; i < 36; i++ {
		city := cities[i%len(cities)]
		x1 := fmt.Sprintf("%.2f", float64(i)*0.4)
		x2 := fmt.Sprintf("%.2f", float64(i%6)*1.2)
		if i%11 == 0 {
			x2 = "NA"
		}
		y := 0
		if i >= 18 {
			y = 1
		}
		sb.WriteString(fmt.Sprintf("ID_%03d,%s,%s,%s,%d\n", i, city, x1, x2, y))
	}
	if err := os.WriteFile(dirtyCSV, []byte(sb.String()), 0o644); err != nil {
		t.Fatal(err)
	}

	workDir := filepath.Join(dir, "autotune_work")
	bestModel := filepath.Join(dir, "best.leaves.json")
	reportPath := filepath.Join(dir, "autotune_report.json")
	if err := cmdAutotune([]string{
		"--data", dirtyCSV,
		"--budget", "3",
		"--rounds", "8",
		"--cv", "2",
		"--out-dir", workDir,
		"--out-model", bestModel,
		"--metrics", reportPath,
	}); err != nil {
		t.Fatalf("cmdAutotune failed: %v", err)
	}

	if _, err := os.Stat(bestModel); err != nil {
		t.Fatalf("expected best model %s to exist: %v", bestModel, err)
	}
	var rep map[string]any
	mustJSON(t, reportPath, &rep)
	if rep["runs_executed"].(float64) != 3 {
		t.Fatalf("expected runs_executed=3, got %v", rep["runs_executed"])
	}
	adapt, _ := rep["auto_adaptations"].(map[string]any)
	if adapt == nil || adapt["auto_encode"] != true || adapt["na_policy"] != "nan" {
		t.Fatalf("expected auto_adaptations to enable auto_encode and na_policy=nan, got %v", adapt)
	}
}

// TestAA5_PublishVerificationAndSchemaSync 验证 WP-44 / DoD AA5：
// leaves publish 自动产出 manifest.verification 跨格式回读推理 parity 证明，且 schema 支持 --subcommand autotune。
func TestAA5_PublishVerificationAndSchemaSync(t *testing.T) {
	dir := t.TempDir()
	csvPath := filepath.Join(dir, "reg.csv")
	var sb strings.Builder
	sb.WriteString("a,b,label\n")
	for i := 0; i < 24; i++ {
		a := float64(i) * 0.5
		b := float64(i%4) * 1.2
		sb.WriteString(fmt.Sprintf("%.2f,%.2f,%.4f\n", a, b, a+b))
	}
	if err := os.WriteFile(csvPath, []byte(sb.String()), 0o644); err != nil {
		t.Fatal(err)
	}

	modelPath := filepath.Join(dir, "m.leaves.json")
	metricsPath := filepath.Join(dir, "m.json")
	if err := cmdTrain([]string{
		"--data", csvPath, "--objective", "reg:squarederror",
		"--rounds", "10", "--depth", "3",
		"--out-model", modelPath, "--metrics", metricsPath,
	}); err != nil {
		t.Fatalf("train: %v", err)
	}

	relDir := filepath.Join(dir, "rel")
	if err := cmdPublish([]string{
		"--model", modelPath, "--out-dir", relDir,
		"--export-xgb", "--export-onnx", "--quantize",
		"--data", csvPath, "--metrics", metricsPath,
	}); err != nil {
		t.Fatalf("publish: %v", err)
	}

	var manifest map[string]any
	mustJSON(t, filepath.Join(relDir, "manifest.json"), &manifest)
	ver, ok := manifest["verification"].(map[string]any)
	if !ok || ver == nil {
		t.Fatalf("expected verification object in manifest.json, got %v", manifest["verification"])
	}
	if ver["verified"] != true || ver["main_load_ok"] != true || ver["xgb_load_ok"] != true || ver["onnx_load_ok"] != true || ver["quant_load_ok"] != true {
		t.Fatalf("expected all verification flags true, got %v", ver)
	}
	if ver["xgb_parity_max_diff"].(float64) > 1e-4 || ver["onnx_parity_max_diff"].(float64) > 1e-4 {
		t.Fatalf("expected 0 parity diff for xgb/onnx export, got %v", ver)
	}

	schemaOut := filepath.Join(dir, "schema_autotune.json")
	if err := cmdSchema([]string{
		"--subcommand", "autotune",
		"--metrics", schemaOut,
	}); err != nil {
		t.Fatalf("schema --subcommand autotune: %v", err)
	}
	var sdoc map[string]any
	mustJSON(t, schemaOut, &sdoc)
	if sdoc["command"] != "autotune" {
		t.Fatalf("expected command=autotune in schema output, got %v", sdoc["command"])
	}
}

// TestAZ1_PredictLinkTransformsAndMetricsSummary 验证 WP-45 / DoD AZ1：
//  1. count:poisson / reg:tweedie 在 predict JSONL/CSV 下自动应用 exp(margin) 变换输出 prediction；
//  2. binary:logistic 在 CSV 下输出 sigmoid 概率且 --metrics 写出完整推理摘要（pred_min/pred_max/pred_mean/positive_rate）。
func TestAZ1_PredictLinkTransformsAndMetricsSummary(t *testing.T) {
	dir := t.TempDir()
	// 1) Poisson count 数据
	poisPath := filepath.Join(dir, "poisson.csv")
	var sb strings.Builder
	sb.WriteString("x1,x2,label\n")
	for i := 0; i < 30; i++ {
		x1 := float64(i%5) * 0.4
		x2 := float64(i%3) * 0.3
		cnt := math.Round(math.Exp(0.5*x1 + 0.3*x2))
		sb.WriteString(fmt.Sprintf("%.2f,%.2f,%.0f\n", x1, x2, cnt))
	}
	if err := os.WriteFile(poisPath, []byte(sb.String()), 0o644); err != nil {
		t.Fatal(err)
	}

	poisModel := filepath.Join(dir, "pois.leaves.json")
	if err := cmdTrain([]string{
		"--data", poisPath, "--objective", "count:poisson",
		"--rounds", "10", "--depth", "3",
		"--out-model", poisModel, "--metrics", filepath.Join(dir, "pois_train.json"),
	}); err != nil {
		t.Fatalf("train poisson: %v", err)
	}

	poisPredJSONL := filepath.Join(dir, "pois_pred.jsonl")
	poisPredMet := filepath.Join(dir, "pois_pred_metrics.json")
	if err := cmdPredict([]string{
		"--model", poisModel, "--data", poisPath,
		"--out", poisPredJSONL, "--metrics", poisPredMet,
	}); err != nil {
		t.Fatalf("predict poisson jsonl: %v", err)
	}
	rawJSONL, err := os.ReadFile(poisPredJSONL)
	if err != nil {
		t.Fatal(err)
	}
	firstLine := strings.Split(strings.TrimSpace(string(rawJSONL)), "\n")[0]
	var rowRec map[string]any
	if err := json.Unmarshal([]byte(firstLine), &rowRec); err != nil {
		t.Fatal(err)
	}
	marginVal, okM := rowRec["margin"].(float64)
	predVal, okP := rowRec["prediction"].(float64)
	if !okM || !okP || math.Abs(predVal-math.Exp(marginVal)) > 1e-6 {
		t.Fatalf("expected prediction == exp(margin), got margin=%v prediction=%v", rowRec["margin"], rowRec["prediction"])
	}

	var pMet map[string]any
	mustJSON(t, poisPredMet, &pMet)
	if int(pMet["n_rows"].(float64)) != 30 || pMet["pred_min"].(float64) <= 0 {
		t.Fatalf("expected n_rows=30 and pred_min>0 for poisson, got %v", pMet)
	}

	// 2) Binary classification CSV + --metrics
	binPath := filepath.Join(dir, "bin.csv")
	sb.Reset()
	sb.WriteString("x1,x2,label\n")
	for i := 0; i < 30; i++ {
		x1 := float64(i) * 0.2
		x2 := float64(30-i) * 0.1
		lbl := 0
		if i >= 15 {
			lbl = 1
		}
		sb.WriteString(fmt.Sprintf("%.2f,%.2f,%d\n", x1, x2, lbl))
	}
	if err := os.WriteFile(binPath, []byte(sb.String()), 0o644); err != nil {
		t.Fatal(err)
	}
	binModel := filepath.Join(dir, "bin.leaves.json")
	if err := cmdTrain([]string{
		"--data", binPath, "--objective", "binary:logistic",
		"--rounds", "10", "--depth", "3",
		"--out-model", binModel, "--metrics", filepath.Join(dir, "bin_train.json"),
	}); err != nil {
		t.Fatalf("train binary: %v", err)
	}
	binCSV := filepath.Join(dir, "bin_pred.csv")
	binMet := filepath.Join(dir, "bin_pred_metrics.json")
	if err := cmdPredict([]string{
		"--model", binModel, "--data", binPath,
		"--format", "csv", "--out", binCSV, "--metrics", binMet,
	}); err != nil {
		t.Fatalf("predict binary csv: %v", err)
	}
	csvRaw, err := os.ReadFile(binCSV)
	if err != nil {
		t.Fatal(err)
	}
	csvLines := strings.Split(strings.TrimSpace(string(csvRaw)), "\n")
	if len(csvLines) != 31 || strings.TrimSpace(csvLines[0]) != "prediction" {
		t.Fatalf("unexpected binary csv lines: %d header=%q", len(csvLines), csvLines[0])
	}
	for _, l := range csvLines[1:] {
		v, perr := strconv.ParseFloat(strings.TrimSpace(l), 64)
		if perr != nil || v <= 0 || v >= 1 {
			t.Fatalf("expected probability in (0,1) for binary csv output, got %q", l)
		}
	}
	var bMet map[string]any
	mustJSON(t, binMet, &bMet)
	if _, ok := bMet["positive_rate"].(float64); !ok {
		t.Fatalf("expected positive_rate in binary predict metrics, got %v", bMet)
	}
}

// TestAZ2_InspectDeepProfileAndQuantOverlay 验证 WP-46 / DoD AZ2：
// leaves inspect 支持深度森林结构统计（max_depth_observed/mean_leaves_per_tree/unused_features）与 .quant.json 量化侧车。
func TestAZ2_InspectDeepProfileAndQuantOverlay(t *testing.T) {
	dir := t.TempDir()
	csvPath := filepath.Join(dir, "feat_unused.csv")
	var sb strings.Builder
	sb.WriteString("signal_a,unused_const,signal_b,label\n")
	for i := 0; i < 32; i++ {
		a := float64(i) * 0.5
		b := float64(i%4) * 1.1
		sb.WriteString(fmt.Sprintf("%.2f,0.00,%.2f,%.4f\n", a, b, 2*a+b))
	}
	if err := os.WriteFile(csvPath, []byte(sb.String()), 0o644); err != nil {
		t.Fatal(err)
	}

	modelPath := filepath.Join(dir, "m.leaves.json")
	metPath := filepath.Join(dir, "m.json")
	if err := cmdTrain([]string{
		"--data", csvPath, "--objective", "reg:squarederror",
		"--rounds", "8", "--depth", "3",
		"--out-model", modelPath, "--metrics", metPath,
	}); err != nil {
		t.Fatalf("train: %v", err)
	}

	relDir := filepath.Join(dir, "rel")
	if err := cmdPublish([]string{
		"--model", modelPath, "--out-dir", relDir,
		"--quantize", "--data", csvPath, "--metrics", metPath,
	}); err != nil {
		t.Fatalf("publish quant: %v", err)
	}

	quantPath := filepath.Join(relDir, "model.quant.json")
	inspOut := filepath.Join(dir, "inspect_quant.json")
	if err := cmdInspect([]string{
		"--model", quantPath, "--metrics", inspOut,
	}); err != nil {
		t.Fatalf("inspect .quant.json: %v", err)
	}

	var idoc map[string]any
	mustJSON(t, inspOut, &idoc)
	if idoc["quantized"] != true || idoc["format"] != "leaves.quant.json" {
		t.Fatalf("expected quantized=true and format=leaves.quant.json, got %v", idoc)
	}
	if idoc["max_depth_observed"].(float64) <= 0 || idoc["mean_leaves_per_tree"].(float64) <= 1 {
		t.Fatalf("expected positive depth and leaves stats, got %v", idoc)
	}
	unused, ok := idoc["unused_features"].([]any)
	if !ok || len(unused) == 0 || unused[0] != "unused_const" {
		t.Fatalf("expected unused_features=[unused_const], got %v", idoc["unused_features"])
	}
}

// TestAZ3_SplitNAPolicyNaNAndDriftReport 验证 WP-47 / DoD AZ3：
// leaves split --na-policy nan 原样保留含缺失特征的样本行并统计 missing_cells、label_means、max_label_drift、leakage_ok。
func TestAZ3_SplitNAPolicyNaNAndDriftReport(t *testing.T) {
	dir := t.TempDir()
	dirtyCSV := filepath.Join(dir, "dirty.csv")
	var sb strings.Builder
	sb.WriteString("f1,f2,label\n")
	for i := 0; i < 40; i++ {
		f1 := fmt.Sprintf("%.2f", float64(i)*0.3)
		if i%5 == 0 {
			f1 = "" // 缺失单元格
		}
		f2 := fmt.Sprintf("%.2f", float64(40-i)*0.2)
		lbl := i % 2
		sb.WriteString(fmt.Sprintf("%s,%s,%d\n", f1, f2, lbl))
	}
	if err := os.WriteFile(dirtyCSV, []byte(sb.String()), 0o644); err != nil {
		t.Fatal(err)
	}

	splitDir := filepath.Join(dir, "splits")
	splitMet := filepath.Join(dir, "split_report.json")
	if err := cmdSplit([]string{
		"--data", dirtyCSV,
		"--out-dir", splitDir,
		"--na-policy", "nan",
		"--ratios", "0.7,0.3",
		"--metrics", splitMet,
	}); err != nil {
		t.Fatalf("split --na-policy nan: %v", err)
	}

	var rep map[string]any
	mustJSON(t, splitMet, &rep)
	if int(rep["total_rows"].(float64)) != 40 {
		t.Fatalf("expected all 40 rows preserved with --na-policy nan, got %v", rep["total_rows"])
	}
	if int(rep["missing_cells"].(float64)) != 8 {
		t.Fatalf("expected 8 missing_cells recorded, got %v", rep["missing_cells"])
	}
	if rep["leakage_ok"] != true {
		t.Fatalf("expected leakage_ok=true, got %v", rep["leakage_ok"])
	}
	lMeans, ok := rep["label_means"].(map[string]any)
	if !ok || lMeans["train"] == nil || lMeans["val"] == nil {
		t.Fatalf("expected label_means for train and val, got %v", rep["label_means"])
	}
	if rep["max_label_drift"].(float64) > 0.1 {
		t.Fatalf("expected stratified split max_label_drift <= 0.1, got %v", rep["max_label_drift"])
	}

	// 验证切分出的 train.csv 可直接喂给 leaves train --na-policy nan
	trainPath := rep["train_path"].(string)
	valPath := rep["val_path"].(string)
	if err := cmdTrain([]string{
		"--data", trainPath, "--val", valPath,
		"--objective", "binary:logistic", "--na-policy", "nan",
		"--rounds", "6", "--depth", "3",
		"--metrics", filepath.Join(dir, "train_after_split.json"),
	}); err != nil {
		t.Fatalf("train after split --na-policy nan: %v", err)
	}
}

// TestAZ4_LessonsMetricsAndAutotuneDistillation 验证 WP-48 / DoD AZ4：
//  1. leaves lessons add|search|list 支持 --metrics PATH 输出结构化 JSON；
//  2. leaves autotune 自动召回 prior_lessons 并在调优提升后自动将获胜超参沉淀为 distilled_lesson。
func TestAZ4_LessonsMetricsAndAutotuneDistillation(t *testing.T) {
	dir := t.TempDir()
	lessonsFile := filepath.Join(dir, "lessons.jsonl")
	t.Setenv("LEAVES_LESSONS_PATH", lessonsFile)

	addMet := filepath.Join(dir, "lesson_add.json")
	if err := cmdLessons([]string{
		"add",
		"--task", "housing_reg",
		"--lesson", "高噪声回归适当提高 lambda=2.0 可降低过拟合",
		"--evidence", "rmse 0.42 -> 0.35",
		"--tag", "reg:squarederror,rmse",
		"--metrics", addMet,
	}); err != nil {
		t.Fatalf("lessons add --metrics: %v", err)
	}
	var addDoc map[string]any
	mustJSON(t, addMet, &addDoc)
	if int(addDoc["count"].(float64)) != 1 {
		t.Fatalf("expected count=1 in lessons add metrics, got %v", addDoc)
	}

	// 构造一个基线容易欠拟合/过拟合而第2轮变异能改进的数据集，跑 autotune
	dataPath := filepath.Join(dir, "housing_reg.csv")
	var sb strings.Builder
	sb.WriteString("x1,x2,x3,label\n")
	for i := 0; i < 48; i++ {
		x1 := float64(i) * 0.15
		x2 := float64(i%6) * 0.4
		x3 := float64((i*3)%7) * 0.25
		y := 1.8*x1 - 0.9*x2 + 0.5*x3
		sb.WriteString(fmt.Sprintf("%.3f,%.3f,%.3f,%.4f\n", x1, x2, x3, y))
	}
	if err := os.WriteFile(dataPath, []byte(sb.String()), 0o644); err != nil {
		t.Fatal(err)
	}

	tuneDir := filepath.Join(dir, "tune")
	tuneReport := filepath.Join(dir, "tune_report.json")
	if err := cmdAutotune([]string{
		"--data", dataPath,
		"--task", "housing_reg",
		"--rounds", "6",
		"--budget", "3",
		"--cv", "2",
		"--out-dir", tuneDir,
		"--metrics", tuneReport,
	}); err != nil {
		t.Fatalf("autotune with lessons: %v", err)
	}

	var rep map[string]any
	mustJSON(t, tuneReport, &rep)
	priors, ok := rep["prior_lessons"].([]any)
	if !ok || len(priors) == 0 {
		t.Fatalf("expected prior_lessons recalled in autotune report, got %v", rep["prior_lessons"])
	}

	listMet := filepath.Join(dir, "lesson_list.json")
	if err := cmdLessons([]string{
		"list", "--task", "housing_reg", "--metrics", listMet,
	}); err != nil {
		t.Fatalf("lessons list --metrics: %v", err)
	}
	var listDoc map[string]any
	mustJSON(t, listMet, &listDoc)
	if rep["improvement_pct"].(float64) > 0 {
		if rep["distilled_lesson"] == nil {
			t.Fatalf("expected distilled_lesson when improvement_pct > 0, got %v", rep)
		}
		if int(listDoc["count"].(float64)) < 2 {
			t.Fatalf("expected at least 2 lessons in memory bank after distillation, got %v", listDoc)
		}
	}
}

// TestAZ5_SchemaAndDocsSync 验证 WP-49 / DoD AZ5：
// leaves schema 覆盖 13 个子命令且包含新增输出契约字段。
func TestAZ5_SchemaAndDocsSync(t *testing.T) {
	dir := t.TempDir()
	schemaPath := filepath.Join(dir, "full_schema.json")
	if err := cmdSchema([]string{"--metrics", schemaPath}); err != nil {
		t.Fatalf("cmdSchema: %v", err)
	}
	var doc map[string]any
	mustJSON(t, schemaPath, &doc)
	cmds, ok := doc["commands"].(map[string]any)
	if !ok || len(cmds) < 12 {
		t.Fatalf("expected >=12 commands in schema, got %v", len(cmds))
	}
	for _, requiredCmd := range []string{"sniff", "split", "train", "autotune", "runs", "diagnose", "eval", "compare", "predict", "inspect", "explain", "publish", "lessons"} {
		if _, exists := cmds[requiredCmd]; !exists {
			t.Fatalf("schema missing command %q", requiredCmd)
		}
	}
}
