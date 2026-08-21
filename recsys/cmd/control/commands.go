package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"time"

	"github.com/linkerlin/leaves/v2/recsys"
	"github.com/linkerlin/leaves/v2/recsys/contract"
	"github.com/linkerlin/leaves/v2/recsys/deal"
	"github.com/linkerlin/leaves/v2/recsys/eval"
	"github.com/linkerlin/leaves/v2/recsys/ledger"
	"github.com/linkerlin/leaves/v2/recsys/monitor"
	"github.com/linkerlin/leaves/v2/recsys/release"
	"github.com/linkerlin/leaves/v2/recsys/replay"
	"github.com/linkerlin/leaves/v2/recsys/split"
	"github.com/linkerlin/leaves/v2/recsys/tsvio"
)

// newFlagSet 只建 FlagSet（命令先注册 flag 再 parseFlags）。
func newFlagSet(name string) *flag.FlagSet {
	fs := flag.NewFlagSet(name, flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	return fs
}

// parseFlags 解析并吞掉默认输出；错误转为 usage 错误。
func parseFlags(fs *flag.FlagSet, args []string) error {
	if err := fs.Parse(args); err != nil {
		return usageErr("flag 解析失败: %v", err)
	}
	return nil
}

func parseTime(flagName, v string) (time.Time, error) {
	t, err := time.Parse(time.RFC3339, v)
	if err != nil {
		return time.Time{}, usageErr("%s 需 RFC3339（如 2026-08-20T10:00:00Z）: %v", flagName, err)
	}
	return t.UTC(), nil
}

// ── snapshot ────────────────────────────────────────────────

func cmdSnapshot(args []string) error {
	fs := newFlagSet("snapshot")
	ws := fs.String("workspace", "", "recsys 工作区根目录")
	out := fs.String("out", "snapshot.json", "输出 snapshot.json")
	id := fs.String("snapshot-id", "", "快照 ID")
	purpose := fs.String("purpose", "", "train|eval|release")
	createdAt := fs.String("created-at", "", "RFC3339；默认 now UTC")
	start := fs.String("time-start", "", "RFC3339；与 -time-end 成对")
	end := fs.String("time-end", "", "RFC3339")
	legacy := fs.Bool("legacy", false, "旧四元数据显式导入（跳过特征指纹校验）")
	if err := parseFlags(fs, args); err != nil {
		return err
	}

	if *id == "" || *purpose == "" {
		return usageErr("-snapshot-id 与 -purpose 必填")
	}
	if *ws == "" {
		return usageErr("-workspace 必填")
	}
	w := recsys.Workspace{Root: *ws}
	snap := &contract.DatasetSnapshot{SnapshotID: *id, SchemaVersion: contract.SchemaVersion, Purpose: *purpose}
	if *createdAt != "" {
		t, err := parseTime("-created-at", *createdAt)
		if err != nil {
			return err
		}
		snap.CreatedAt = t
	} else {
		snap.CreatedAt = time.Now().UTC()
	}
	snap.LegacySnapshot = *legacy
	if *start != "" || *end != "" {
		if *start == "" || *end == "" {
			return usageErr("-time-start 与 -time-end 必须成对")
		}
		ts, err := parseTime("-time-start", *start)
		if err != nil {
			return err
		}
		te, err := parseTime("-time-end", *end)
		if err != nil {
			return err
		}
		snap.TimeRange = contract.TimeRange{Start: ts, End: te}
	}
	for _, p := range []string{w.SamplesTrain(), w.SamplesTest(), w.ItemsCatalog()} {
		if _, err := os.Stat(p); err != nil {
			return fmt.Errorf("缺少工作区工件 %s（先跑四段流水线）", p)
		}
		h, err := contract.HashFile(p)
		if err != nil {
			return err
		}
		rel, err := filepath.Rel(w.Root, p)
		if err != nil {
			return err
		}
		snap.InputFiles = append(snap.InputFiles, contract.FileRef{Path: filepath.ToSlash(rel), SHA256: h})
	}
	featNames, _, err := tsvio.ReadCatalog(w.ItemsCatalog())
	if err != nil {
		return err
	}
	snap.FeatureSchemaHash = contract.FeatureSchemaHash(featNames, nil)
	if err := contract.ValidateSnapshot(snap); err != nil {
		return err
	}
	if err := tsvio.WriteJSON(*out, snap); err != nil {
		return err
	}
	fmt.Printf("快照 %s 已写 %s（%d 输入文件指纹）\n", snap.SnapshotID, *out, len(snap.InputFiles))
	return nil
}

// ── split ───────────────────────────────────────────────────

func cmdSplit(args []string) error {
	fs := newFlagSet("split")
	eventsPath := fs.String("events", "", "InteractionEvent JSONL")
	trainEnd := fs.String("train-end", "", "训练窗口终点（排他）")
	valStart := fs.String("val-start", "", "验证窗口起点")
	testStart := fs.String("test-start", "", "测试窗口起点")
	outDir := fs.String("out-dir", "", "输出目录（train/val/test.jsonl + split_report.json）")
	if err := parseFlags(fs, args); err != nil {
		return err
	}

	for name, v := range map[string]string{"-events": *eventsPath, "-train-end": *trainEnd, "-val-start": *valStart, "-test-start": *testStart, "-out-dir": *outDir} {
		if v == "" {
			return usageErr("%s 必填", name)
		}
	}
	te, err := parseTime("-train-end", *trainEnd)
	if err != nil {
		return err
	}
	vs, err := parseTime("-val-start", *valStart)
	if err != nil {
		return err
	}
	ts, err := parseTime("-test-start", *testStart)
	if err != nil {
		return err
	}
	events, err := contract.ReadJSONL[contract.InteractionEvent](*eventsPath)
	if err != nil {
		return err
	}
	train, val, test, err := split.Split(events, split.TimeConfig{TrainEnd: te, ValStart: vs, TestStart: ts})
	if err != nil {
		return err
	}
	if err := split.CheckLeakage(train, vs); err != nil {
		return err
	}
	if err := os.MkdirAll(*outDir, 0o755); err != nil {
		return err
	}
	for name, rows := range map[string][]contract.InteractionEvent{"train.jsonl": train, "val.jsonl": val, "test.jsonl": test} {
		if err := contract.WriteJSONL(filepath.Join(*outDir, name), rows); err != nil {
			return err
		}
	}
	report := map[string]any{
		"train": len(train), "val": len(val), "test": len(test),
		"gap_dropped": len(events) - len(train) - len(val) - len(test),
		"leakage_ok":  true,
	}
	if err := tsvio.WriteJSON(filepath.Join(*outDir, "split_report.json"), report); err != nil {
		return err
	}
	fmt.Printf("时间切分: train=%d val=%d test=%d gap=%d（泄漏检查通过）\n",
		len(train), len(val), len(test), len(events)-len(train)-len(val)-len(test))
	return nil
}

// ── eval ────────────────────────────────────────────────────

func cmdEval(args []string) error {
	fs := newFlagSet("eval")
	ws := fs.String("workspace", "", "recsys 工作区根目录")
	thresholdsPath := fs.String("thresholds", "", "eval.Threshold JSON 数组；缺省 → exploratory")
	out := fs.String("out", "evaluation.json", "输出 evaluation.json")
	recallK := fs.Int("recall-k", 100, "Recall@K")
	ndcgK := fs.Int("ndcg-k", 10, "NDCG/MAP@K")
	deckSize := fs.Int("deck-size", 10, "发牌 deck 大小")
	maxSameTag := fs.Int("max-same-tag", 3, "发牌 Tag 控重上限")
	leakCount := fs.Int("leak-count", 0, "数据层泄漏事件数（先跑 split）")
	eventCount := fs.Int("event-count", 0, "数据层事件总数")
	if err := parseFlags(fs, args); err != nil {
		return err
	}

	if *ws == "" {
		return usageErr("-workspace 必填")
	}
	w := recsys.Workspace{Root: *ws}
	testSamples, err := tsvio.ReadInteractions(w.SamplesTest())
	if err != nil {
		return err
	}
	scored, err := tsvio.ReadManifestJSONL(w.RankTestScored())
	if err != nil {
		return err
	}
	_, catalog, err := tsvio.ReadCatalog(w.ItemsCatalog())
	if err != nil {
		return err
	}
	deals, err := tsvio.ReadDeal(w.DealTest())
	if err != nil {
		return err
	}
	relevant, ranked, groups := eval.RankViews(testSamples, scored)

	var ths []eval.Threshold
	if *thresholdsPath != "" {
		if err := readJSON(*thresholdsPath, &ths); err != nil {
			return err
		}
	}
	cfg := eval.Config{
		RecallK: *recallK, NDCGK: *ndcgK, DeckSize: *deckSize, MaxSameTag: *maxSameTag,
		Thresholds: ths,
	}
	report := eval.Evaluate(cfg, eval.Inputs{
		Relevant: relevant, Ranked: ranked, CatalogSize: len(catalog),
		Groups: groups, Deals: deals,
		DataLeakCount: *leakCount, DataEventCount: *eventCount,
	})
	if err := eval.WriteReport(*out, report); err != nil {
		return err
	}
	fmt.Printf("评估 %s: purpose=%s status=%s（%d 指标, %d 门禁）\n",
		*out, report.Purpose, report.Status, len(report.Metrics), len(report.Gates))
	return nil
}

// ── from-deal ───────────────────────────────────────────────

func cmdFromDeal(args []string) error {
	fs := newFlagSet("from-deal")
	ws := fs.String("workspace", "", "recsys 工作区根目录")
	ledgerPath := fs.String("ledger", "ledger.jsonl", "决策账本 JSONL（追加写）")
	modelVersion := fs.String("model-version", "", "模型版本")
	policyVersion := fs.String("policy-version", "", "策略版本")
	occurredAt := fs.String("occurred-at", "", "决策时间 RFC3339")
	featureHash := fs.String("feature-schema-hash", "", "特征指纹；缺省从 items.tsv 推导")
	candidateSetID := fs.String("candidate-set-id", "", "候选集 ID；缺省 cs-<工作区名>")
	if err := parseFlags(fs, args); err != nil {
		return err
	}

	if *ws == "" || *modelVersion == "" || *policyVersion == "" || *occurredAt == "" {
		return usageErr("-workspace/-model-version/-policy-version/-occurred-at 必填")
	}
	at, err := parseTime("-occurred-at", *occurredAt)
	if err != nil {
		return err
	}
	w := recsys.Workspace{Root: *ws}
	dealRows, err := tsvio.ReadDeal(w.DealTest())
	if err != nil {
		return err
	}
	logs, err := deal.ReadLog(w.DealLog())
	if err != nil {
		return err
	}
	if *featureHash == "" {
		featNames, _, err := tsvio.ReadCatalog(w.ItemsCatalog())
		if err != nil {
			return err
		}
		*featureHash = contract.FeatureSchemaHash(featNames, nil)
	}
	if *candidateSetID == "" {
		*candidateSetID = "cs-" + filepath.Base(w.Root)
	}

	lg, err := ledger.Open(*ledgerPath)
	if err != nil {
		return err
	}
	deckByUser, logByUser := groupDeal(dealRows, logs)
	users := sortedKeys(deckByUser)
	for _, u := range users {
		ev, err := ledger.DecisionFromDeal(u, deckByUser[u], logByUser[u], ledger.DecisionMeta{
			DecisionID: "dec-" + u, RequestID: "req-" + u, SubjectID: u,
			OccurredAt: at, ModelVersion: *modelVersion,
			FeatureSchemaHash: *featureHash, CandidateSetID: *candidateSetID,
			PolicyVersion: *policyVersion,
		})
		if err != nil {
			return err
		}
		if err := lg.AppendDecision(ev); err != nil {
			return err
		}
	}
	fmt.Printf("账本 %s 追加 %d 条决策（%d 用户）\n", *ledgerPath, len(users), len(users))
	return nil
}

// ── append-exposure / append-feedback ───────────────────────

func cmdAppend(args []string, kind string) error {
	fs := newFlagSet("append-" + kind)
	ledgerPath := fs.String("ledger", "ledger.jsonl", "账本 JSONL")
	in := fs.String("in", "", "事件 JSONL 输入")
	if err := parseFlags(fs, args); err != nil {
		return err
	}
	if *in == "" {
		return usageErr("-in 必填")
	}
	lg, err := ledger.Open(*ledgerPath)
	if err != nil {
		return err
	}
	switch kind {
	case "exposure":
		rows, err := contract.ReadJSONL[contract.ExposureEvent](*in)
		if err != nil {
			return err
		}
		for i := range rows {
			if err := lg.AppendExposure(rows[i]); err != nil {
				return fmt.Errorf("第 %d 行: %w", i+1, err)
			}
		}
		fmt.Printf("追加 %d 条曝光\n", len(rows))
	case "feedback":
		rows, err := contract.ReadJSONL[contract.FeedbackEvent](*in)
		if err != nil {
			return err
		}
		for i := range rows {
			if err := lg.AppendFeedback(rows[i]); err != nil {
				return fmt.Errorf("第 %d 行: %w", i+1, err)
			}
		}
		fmt.Printf("追加 %d 条反馈\n", len(rows))
	}
	return nil
}

// ── replay ──────────────────────────────────────────────────

func cmdReplay(args []string) error {
	fs := newFlagSet("replay")
	ledgerPath := fs.String("ledger", "ledger.jsonl", "账本 JSONL")
	out := fs.String("out", "samples.jsonl", "输出训练样本 JSONL")
	report := fs.String("report", "replay_report.json", "输出 replay_report.json")
	window := fs.String("window", "24h", "归因窗时长（如 24h、30m）")
	negative := fs.String("negative", "impressed_no_feed", "负样本策略 none|impressed_no_feed")
	positiveThreshold := fs.Float64("positive-threshold", 0, "正样本阈值（0=反馈值>0 即正）")
	if err := parseFlags(fs, args); err != nil {
		return err
	}

	d, err := time.ParseDuration(*window)
	if err != nil {
		return usageErr("-window 需时长（如 24h）: %v", err)
	}
	cfg := replay.Config{AttributionWindow: d, NegativePolicy: *negative}
	if *positiveThreshold != 0 {
		v := *positiveThreshold
		cfg.PositiveThreshold = &v
	}
	lg, err := ledger.Open(*ledgerPath)
	if err != nil {
		return err
	}
	samples, rep, err := replay.BuildSamples(lg, cfg)
	if err != nil {
		return err
	}
	if err := contract.WriteJSONL(*out, samples); err != nil {
		return err
	}
	if err := replay.WriteReport(*report, rep); err != nil {
		return err
	}
	fmt.Printf("回放: 正=%d 负=%d（迟到=%d 孤立=%d）→ %s\n",
		rep.Positives, rep.Negatives, rep.LateFeedback, rep.OrphanFeedback, *out)
	return nil
}

// ── monitor ─────────────────────────────────────────────────

func cmdMonitor(args []string) error {
	fs := newFlagSet("monitor")
	ledgerPath := fs.String("ledger", "ledger.jsonl", "账本 JSONL")
	ws := fs.String("workspace", "", "recsys 工作区根目录")
	winStart := fs.String("window-start", "", "窗口起点 RFC3339")
	winEnd := fs.String("window-end", "", "窗口终点 RFC3339")
	thresholdsPath := fs.String("thresholds", "", "monitor.Threshold JSON 数组")
	triggersPath := fs.String("triggers", "", "monitor.Trigger JSON 数组")
	out := fs.String("out", "monitor_report.json", "输出 monitor_report.json")
	fired := fs.String("fired", "fired.jsonl", "触发动作输出 JSONL")
	deckSize := fs.Int("deck-size", 10, "发牌 deck 大小")
	maxSameTag := fs.Int("max-same-tag", 3, "发牌 Tag 控重上限")
	if err := parseFlags(fs, args); err != nil {
		return err
	}

	if *ws == "" || *winStart == "" || *winEnd == "" {
		return usageErr("-workspace/-window-start/-window-end 必填")
	}
	start, err := parseTime("-window-start", *winStart)
	if err != nil {
		return err
	}
	end, err := parseTime("-window-end", *winEnd)
	if err != nil {
		return err
	}
	w := recsys.Workspace{Root: *ws}
	deals, err := tsvio.ReadDeal(w.DealTest())
	if err != nil {
		return err
	}
	_, catalog, err := tsvio.ReadCatalog(w.ItemsCatalog())
	if err != nil {
		return err
	}
	lg, err := ledger.Open(*ledgerPath)
	if err != nil {
		return err
	}
	var ths []monitor.Threshold
	if *thresholdsPath != "" {
		if err := readJSON(*thresholdsPath, &ths); err != nil {
			return err
		}
	}
	rep := monitor.BuildReport(lg, start, end, deals, *deckSize, *maxSameTag, len(catalog), ths)
	if err := monitor.WriteReport(*out, rep); err != nil {
		return err
	}
	if *triggersPath != "" {
		var rules []monitor.Trigger
		if err := readJSON(*triggersPath, &rules); err != nil {
			return err
		}
		ts, err := monitor.NewTriggerSet(rules)
		if err != nil {
			return err
		}
		firedEvents := ts.Evaluate(rep, end)
		if err := contract.WriteJSONL(*fired, firedEvents); err != nil {
			return err
		}
		fmt.Printf("触发器: %d 条触发 → %s\n", len(firedEvents), *fired)
	}
	fmt.Printf("监控 %s: overall=%s（%d 指标）\n", *out, rep.Overall, len(rep.Metrics))
	return nil
}

// ── release ─────────────────────────────────────────────────

// evalReportFile evaluation.json 的读取视图。
type evalReportFile struct {
	Metrics map[string]float64    `json:"metrics"`
	Gates   []contract.GateResult `json:"gates"`
}

func cmdRelease(args []string) error {
	fs := newFlagSet("release")
	statePath := fs.String("state", "release_state.json", "状态文件")
	action := fs.String("action", "", "candidate|approve|confirm-promote|observe|retrain|rollback|retire|status")
	releaseID := fs.String("release-id", "", "candidate 用：release ID")
	evaluation := fs.String("evaluation", "evaluation.json", "candidate 用：evaluation.json")
	model := fs.String("model", "", "candidate 用：模型文件路径（hash 校验）")
	runID := fs.String("run-id", "", "candidate 用：训练 run ID")
	snapshotID := fs.String("snapshot-id", "", "candidate 用：快照 ID")
	policyVersion := fs.String("policy-version", "", "candidate 用：策略版本")
	lastKnownGood := fs.String("last-known-good", "", "candidate 用：回滚锚点 release ID")
	approver := fs.String("approver", "", "approve 用：审批人")
	modelVersion := fs.String("model-version", "", "confirm-promote 用：模型版本")
	reason := fs.String("reason", "", "retrain/rollback/retire 用：原因")
	atFlag := fs.String("at", "", "RFC3339；默认 now UTC")
	if err := parseFlags(fs, args); err != nil {
		return err
	}

	at := time.Now().UTC()
	if *atFlag != "" {
		t, err := parseTime("-at", *atFlag)
		if err != nil {
			return err
		}
		at = t
	}

	var m *release.Machine
	if b, err := os.ReadFile(*statePath); err == nil {
		var ms release.MachineState
		if err := json.Unmarshal(b, &ms); err != nil {
			return fmt.Errorf("状态文件 %s 解析失败: %w", *statePath, err)
		}
		m, err = release.FromState(ms)
		if err != nil {
			return err
		}
	} else if !os.IsNotExist(err) {
		return err
	}

	switch *action {
	case "candidate":
		if m != nil {
			return fmt.Errorf("状态文件 %s 已存在（candidate 需全新 release）", *statePath)
		}
		if *releaseID == "" || *runID == "" || *snapshotID == "" || *policyVersion == "" || *model == "" {
			return usageErr("candidate 需要 -release-id/-run-id/-snapshot-id/-policy-version/-model")
		}
		var er evalReportFile
		if err := readJSON(*evaluation, &er); err != nil {
			return err
		}
		modelHash, err := contract.HashFile(*model)
		if err != nil {
			return err
		}
		ev := contract.ReleaseEvidence{
			ReleaseID: *releaseID, ModelSHA256: modelHash,
			RunID: *runID, SnapshotID: *snapshotID, PolicyVersion: *policyVersion,
			OfflineMetrics: er.Metrics, Gates: er.Gates, CreatedAt: at,
		}
		m = release.NewMachine(*releaseID)
		if *lastKnownGood != "" {
			if err := m.SetLastKnownGood(*lastKnownGood); err != nil {
				return err
			}
		}
		if err := m.ToCandidate(ev, *model, at); err != nil {
			return err
		}
	case "approve":
		if m == nil {
			return usageErr("无状态文件 %s，先跑 candidate", *statePath)
		}
		if err := m.Approve(*approver, at); err != nil {
			return err
		}
	case "confirm-promote":
		if m == nil {
			return usageErr("无状态文件 %s，先跑 candidate", *statePath)
		}
		if err := m.ConfirmPromoted(at); err != nil {
			return err
		}
		ev := m.Evidence()
		req := release.PromoteRequest{
			ReleaseID: m.ReleaseID, ModelVersion: *modelVersion, ModelSHA256: ev.ModelSHA256,
		}
		printJSON("promote_request", req)
	case "observe":
		if m == nil {
			return usageErr("无状态文件 %s，先跑 candidate", *statePath)
		}
		if err := m.Observe(at); err != nil {
			return err
		}
	case "retrain":
		if m == nil {
			return usageErr("无状态文件 %s，先跑 candidate", *statePath)
		}
		if err := m.RequestRetrain(*reason, at); err != nil {
			return err
		}
	case "rollback":
		if m == nil {
			return usageErr("无状态文件 %s，先跑 candidate", *statePath)
		}
		req, err := m.RequestRollback(*reason, at)
		if err != nil {
			return err
		}
		printJSON("rollback_request", req)
	case "retire":
		if m == nil {
			return usageErr("无状态文件 %s，先跑 candidate", *statePath)
		}
		if err := m.Retire(*reason, at); err != nil {
			return err
		}
	case "status":
		if m == nil {
			fmt.Printf("无状态文件 %s（exploratory 之前）\n", *statePath)
			return nil
		}
		printJSON("release_state", m.Export())
		return nil
	default:
		return usageErr("未知 -action %q（candidate|approve|confirm-promote|observe|retrain|rollback|retire|status）", *action)
	}

	st := m.Export()
	if err := tsvio.WriteJSON(*statePath, st); err != nil {
		return err
	}
	fmt.Printf("release %s → %s（状态已写 %s）\n", m.ReleaseID, st.State, *statePath)
	return nil
}

// ── helpers ─────────────────────────────────────────────────

func readJSON(path string, v any) error {
	b, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("读 %s: %w", path, err)
	}
	if err := json.Unmarshal(b, v); err != nil {
		return fmt.Errorf("解析 %s: %w", path, err)
	}
	return nil
}

func printJSON(kind string, v any) {
	b, _ := json.MarshalIndent(v, "", "  ")
	fmt.Printf("%s:\n%s\n", kind, b)
}

func groupDeal(rows []recsys.DealRow, logs []deal.LogEntry) (map[string][]recsys.DealRow, map[string]deal.LogEntry) {
	decks := map[string][]recsys.DealRow{}
	for _, r := range rows {
		decks[r.User] = append(decks[r.User], r)
	}
	ls := map[string]deal.LogEntry{}
	for _, l := range logs {
		ls[l.User] = l
	}
	return decks, ls
}

func sortedKeys(m map[string][]recsys.DealRow) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}
