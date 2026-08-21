package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/linkerlin/leaves/v2/recsys"
	"github.com/linkerlin/leaves/v2/recsys/contract"
	"github.com/linkerlin/leaves/v2/recsys/eval"
	"github.com/linkerlin/leaves/v2/recsys/monitor"
	"github.com/linkerlin/leaves/v2/recsys/pipeline"
	"github.com/linkerlin/leaves/v2/recsys/release"
	"github.com/linkerlin/leaves/v2/recsys/synth"
	"github.com/linkerlin/leaves/v2/recsys/tsvio"
)

func writeJSON(t *testing.T, path string, v any) {
	t.Helper()
	if err := tsvio.WriteJSON(path, v); err != nil {
		t.Fatal(err)
	}
}

func readJSONFile(t *testing.T, path string, v any) {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(b, v); err != nil {
		t.Fatal(err)
	}
}

func timedEvents(t *testing.T, ds synth.Dataset) []contract.InteractionEvent {
	t.Helper()
	byUser := map[string][]recsys.Interaction{}
	var order []string
	for _, r := range ds.Raw {
		if _, ok := byUser[r.User]; !ok {
			order = append(order, r.User)
		}
		byUser[r.User] = append(byUser[r.User], r)
	}
	sort.Strings(order)
	start := time.Date(2026, 8, 1, 0, 0, 0, 0, time.UTC)
	span := 14 * 24 * time.Hour
	var out []contract.InteractionEvent
	for _, u := range order {
		rows := byUser[u]
		for i, r := range rows {
			at := start.Add(time.Duration(float64(span) * float64(i) / float64(len(rows))))
			typ := contract.EventClick
			if i%2 == 1 {
				typ = contract.EventRating
			}
			out = append(out, contract.InteractionEvent{
				EventID: "ev-" + u + "-" + r.Item, OccurredAt: at,
				SubjectID: u, ItemID: r.Item, EventType: typ, Value: r.Score, Source: "cli-test",
			})
		}
	}
	if err := contract.ValidateInteractions(out); err != nil {
		t.Fatal(err)
	}
	return out
}

// TestControlCLIEndToEnd 以 shell 命令函数跑完八段剧本。
func TestControlCLIEndToEnd(t *testing.T) {
	root := t.TempDir()
	w, err := recsys.NewWorkspace(root)
	if err != nil {
		t.Fatal(err)
	}
	cfg := recsys.DefaultSmokeConfig()
	if _, err := pipeline.Run(w, cfg); err != nil {
		t.Fatal(err)
	}
	now := time.Date(2026, 8, 20, 12, 0, 0, 0, time.UTC)

	// 1. snapshot
	snapPath := filepath.Join(root, "snapshot.json")
	if err := cmdSnapshot([]string{
		"-workspace", root, "-out", snapPath, "-snapshot-id", "snap-1", "-purpose", "release",
		"-created-at", now.Format(time.RFC3339),
		"-time-start", "2026-08-01T00:00:00Z", "-time-end", "2026-08-20T00:00:00Z",
	}); err != nil {
		t.Fatal(err)
	}
	var snap contract.DatasetSnapshot
	readJSONFile(t, snapPath, &snap)
	if err := contract.ValidateSnapshot(&snap); err != nil {
		t.Fatal(err)
	}
	if err := contract.VerifyFiles(&snap, root); err != nil {
		t.Fatal(err)
	}

	// 1b. split（合成事件 + 泄漏检查）
	ds, err := synth.Generate(cfg)
	if err != nil {
		t.Fatal(err)
	}
	events := timedEvents(t, ds)
	eventsPath := filepath.Join(root, "events.jsonl")
	if err := contract.WriteJSONL(eventsPath, events); err != nil {
		t.Fatal(err)
	}
	splitDir := filepath.Join(root, "split")
	if err := cmdSplit([]string{
		"-events", eventsPath,
		"-train-end", "2026-08-06T00:00:00Z",
		"-val-start", "2026-08-06T01:00:00Z",
		"-test-start", "2026-08-13T00:00:00Z",
		"-out-dir", splitDir,
	}); err != nil {
		t.Fatal(err)
	}
	var splitRep map[string]any
	readJSONFile(t, filepath.Join(splitDir, "split_report.json"), &splitRep)
	if splitRep["leakage_ok"] != true || splitRep["train"].(float64) == 0 {
		t.Fatalf("split report wrong: %+v", splitRep)
	}

	// 2. eval（三层门禁）
	thresholds := []eval.Threshold{
		{Layer: eval.LayerData, Name: "data_leakage_rate", Max: ptr(0), Level: contract.StatusBlock},
		{Layer: eval.LayerCandidateRank, Name: "recall_at_k", Min: ptr(0.5), Level: contract.StatusBlock},
		{Layer: eval.LayerCandidateRank, Name: "ndcg_at_k", Min: ptr(0.01), Level: contract.StatusBlock},
		{Layer: eval.LayerDeal, Name: "deck_fill_rate", Min: ptr(0.8), Level: contract.StatusBlock},
	}
	writeJSON(t, filepath.Join(root, "thresholds.json"), thresholds)
	evalPath := filepath.Join(root, "evaluation.json")
	if err := cmdEval([]string{
		"-workspace", root, "-thresholds", filepath.Join(root, "thresholds.json"),
		"-out", evalPath, "-recall-k", "100", "-event-count", "200",
	}); err != nil {
		t.Fatal(err)
	}
	var er evalReportFile
	readJSONFile(t, evalPath, &er)
	if len(er.Gates) != 4 || er.Gates[0].Status != contract.StatusOK {
		t.Fatalf("gates wrong: %+v", er.Gates)
	}

	// 3. from-deal → 账本决策
	ledgerPath := filepath.Join(root, "ledger.jsonl")
	if err := cmdFromDeal([]string{
		"-workspace", root, "-ledger", ledgerPath,
		"-model-version", "m-drill", "-policy-version", "deal-v1",
		"-occurred-at", now.Format(time.RFC3339),
	}); err != nil {
		t.Fatal(err)
	}

	// 4. 曝光 + 反馈（从账本决策反推）
	decisions := readDecisions(t, ledgerPath)
	if len(decisions) == 0 {
		t.Fatal("no decisions in ledger")
	}
	first := decisions[0]
	var exposures []contract.ExposureEvent
	for i, it := range first.Items {
		exposures = append(exposures, contract.ExposureEvent{
			ExposureID: "cli-exp-" + first.DecisionID + "-" + it.ItemID,
			DecisionID: first.DecisionID, ItemID: it.ItemID, Position: it.Rank,
			OccurredAt: first.OccurredAt.Add(time.Duration(it.Rank) * time.Second),
			Status:     contract.ExposureShown,
		})
		_ = i
	}
	writeJSONL(t, filepath.Join(root, "exposures.jsonl"), exposures)
	if err := cmdAppend([]string{"-ledger", ledgerPath, "-in", filepath.Join(root, "exposures.jsonl")}, "exposure"); err != nil {
		t.Fatal(err)
	}
	feedback := []contract.FeedbackEvent{{
		EventID: "cli-fb-1", ExposureID: exposures[0].ExposureID,
		OccurredAt: first.OccurredAt.Add(time.Hour), EventType: contract.EventClick, Value: 1,
	}}
	writeJSONL(t, filepath.Join(root, "feedback.jsonl"), feedback)
	if err := cmdAppend([]string{"-ledger", ledgerPath, "-in", filepath.Join(root, "feedback.jsonl")}, "feedback"); err != nil {
		t.Fatal(err)
	}

	// 5. replay（只消费可归因反馈）
	if err := cmdReplay([]string{
		"-ledger", ledgerPath, "-out", filepath.Join(root, "samples.jsonl"),
		"-report", filepath.Join(root, "replay_report.json"), "-window", "24h",
	}); err != nil {
		t.Fatal(err)
	}
	var rep struct {
		Positives int `json:"positives"`
		Negatives int `json:"negatives"`
	}
	readJSONFile(t, filepath.Join(root, "replay_report.json"), &rep)
	if rep.Positives != 1 || rep.Negatives < 1 {
		t.Fatalf("replay wrong: %+v", rep)
	}

	// 6. monitor 健康窗口（触发器规则文件先就位：此时不应触发）
	monTh := []monitor.Threshold{
		{Layer: "deal", Name: "deck_fill_rate", Min: ptr(0.8), Level: contract.StatusBlock},
		{Layer: "data", Name: "orphan_feedback_rate", Max: ptr(0), Level: contract.StatusBlock},
	}
	writeJSON(t, filepath.Join(root, "mon_thresholds.json"), monTh)
	triggers := []monitor.Trigger{{
		Name: "deck-fill-hard", Metric: "deck_fill_rate", Level: contract.StatusBlock,
		Consecutive: 1, Action: monitor.ActionRollback, Cooldown: time.Hour,
	}}
	writeJSON(t, filepath.Join(root, "triggers.json"), triggers)
	monOut := filepath.Join(root, "monitor_report.json")
	if err := cmdMonitor([]string{
		"-ledger", ledgerPath, "-workspace", root,
		"-window-start", now.Format(time.RFC3339), "-window-end", now.Add(2 * time.Hour).Format(time.RFC3339),
		"-thresholds", filepath.Join(root, "mon_thresholds.json"),
		"-triggers", filepath.Join(root, "triggers.json"), "-out", monOut,
		"-fired", filepath.Join(root, "fired.jsonl"),
	}); err != nil {
		t.Fatal(err)
	}
	var monRep struct {
		Overall string `json:"overall"`
	}
	readJSONFile(t, monOut, &monRep)
	if monRep.Overall != "ok" {
		t.Fatalf("healthy monitor: %s", monRep.Overall)
	}
	if err := cmdMonitor([]string{
		"-ledger", ledgerPath, "-workspace", root,
		"-window-start", now.Format(time.RFC3339), "-window-end", now.Add(2 * time.Hour).Format(time.RFC3339),
		"-triggers", filepath.Join(root, "triggers.json"), "-out", monOut,
		"-fired", filepath.Join(root, "fired.jsonl"),
	}); err != nil {
		t.Fatal(err)
	}
	fired, err := contract.ReadJSONL[monitor.Fired](filepath.Join(root, "fired.jsonl"))
	if err != nil {
		t.Fatal(err)
	}
	if len(fired) != 0 {
		t.Fatalf("healthy window must not fire: %+v", fired)
	}

	// 7. release：candidate → approve → confirm-promote → observe
	statePath := filepath.Join(root, "release_state.json")
	if err := cmdRelease([]string{
		"-state", statePath, "-action", "candidate",
		"-release-id", "rel-1", "-evaluation", evalPath, "-model", w.ModelPath(),
		"-run-id", "run-1", "-snapshot-id", "snap-1", "-policy-version", "deal-v1",
		"-last-known-good", "rel-0", "-at", now.Format(time.RFC3339),
	}); err != nil {
		t.Fatal(err)
	}
	if err := cmdRelease([]string{"-state", statePath, "-action", "approve", "-approver", "cli-human", "-at", now.Format(time.RFC3339)}); err != nil {
		t.Fatal(err)
	}
	if err := cmdRelease([]string{"-state", statePath, "-action", "confirm-promote", "-model-version", "m-drill", "-at", now.Format(time.RFC3339)}); err != nil {
		t.Fatal(err)
	}
	if err := cmdRelease([]string{"-state", statePath, "-action", "observe", "-at", now.Format(time.RFC3339)}); err != nil {
		t.Fatal(err)
	}
	var st release.MachineState
	readJSONFile(t, statePath, &st)
	// 锚点保持 candidate 时显式设置的 rel-0（Observe 只在锚点为空时设自身）
	if st.State != release.StateObserving || st.LastKnownGood != "rel-0" || st.Evidence == nil {
		t.Fatalf("state wrong: %+v", st)
	}

	// 8. 退化注入 → 触发器 fired → rollback
	degraded := filepath.Join(root, "deal_degraded.tsv")
	deals, err := tsvio.ReadDeal(w.DealTest())
	if err != nil {
		t.Fatal(err)
	}
	var kept []recsys.DealRow
	for _, d := range deals {
		kept = append(kept, recsys.DealRow{User: d.User, Item: d.Item, Tag: d.Tag, Score: d.Score, Rank: d.Rank})
		if len(kept) == len(deals) { // 保持原状？退化：每用户只留 1 条
			break
		}
	}
	_ = kept
	// 覆写为每用户 1 条
	byUser := map[string]recsys.DealRow{}
	for _, d := range deals {
		byUser[d.User] = d
	}
	var deg []recsys.DealRow
	for _, u := range sortedKeys2(byUser) {
		deg = append(deg, byUser[u])
	}
	if err := tsvio.WriteDeal(degraded, deg); err != nil {
		t.Fatal(err)
	}
	// 用独立工作区（仅 deal 退化，catalog 用原工作区）
	degWS := filepath.Join(root, "ws_degraded")
	if err := os.MkdirAll(filepath.Join(degWS, "deal"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(degWS, "catalog"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(degWS, "deal", "deal_test.tsv"), mustRead(t, degraded), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(degWS, "catalog", "items.tsv"), mustRead(t, w.ItemsCatalog()), 0o644); err != nil {
		t.Fatal(err)
	}
	monDeg := filepath.Join(root, "monitor_deg.jsonl")
	if err := cmdMonitor([]string{
		"-ledger", ledgerPath, "-workspace", degWS,
		"-window-start", now.Format(time.RFC3339), "-window-end", now.Add(3 * time.Hour).Format(time.RFC3339),
		"-thresholds", filepath.Join(root, "mon_thresholds.json"),
		"-triggers", filepath.Join(root, "triggers.json"), "-out", monDeg,
		"-fired", filepath.Join(root, "fired.jsonl"),
	}); err != nil {
		t.Fatal(err)
	}
	fired, err = contract.ReadJSONL[monitor.Fired](filepath.Join(root, "fired.jsonl"))
	if err != nil {
		t.Fatal(err)
	}
	if len(fired) != 1 || fired[0].Action != monitor.ActionRollback {
		t.Fatalf("degraded must fire rollback: %+v", fired)
	}
	if err := cmdRelease([]string{
		"-state", statePath, "-action", "rollback", "-reason", fired[0].Reason, "-at", now.Add(3 * time.Hour).Format(time.RFC3339),
	}); err != nil {
		t.Fatal(err)
	}
	readJSONFile(t, statePath, &st)
	if st.State != release.StateRollbackRequested || st.LastKnownGood != "rel-0" {
		t.Fatalf("rollback state wrong: %+v", st)
	}
}

func TestControlSplitBadOrderFails(t *testing.T) {
	dir := t.TempDir()
	events := []contract.InteractionEvent{
		{EventID: "a", OccurredAt: time.Date(2026, 8, 15, 0, 0, 0, 0, time.UTC), SubjectID: "u1", ItemID: "i1", EventType: contract.EventClick},
	}
	p := filepath.Join(dir, "e.jsonl")
	if err := contract.WriteJSONL(p, events); err != nil {
		t.Fatal(err)
	}
	err := cmdSplit([]string{
		"-events", p, "-train-end", "2026-08-20T00:00:00Z", // 违反 train_end < val_start
		"-val-start", "2026-08-06T01:00:00Z", "-test-start", "2026-08-13T00:00:00Z",
		"-out-dir", filepath.Join(dir, "out"),
	})
	if err == nil || !strings.Contains(err.Error(), "train_end < validation_start") {
		t.Fatalf("want order failure, got %v", err)
	}
	if exitCode(err) != 2 {
		t.Fatalf("validation error: exit=%d", exitCode(err))
	}
}

func TestControlAppendExposureOrphanFails(t *testing.T) {
	dir := t.TempDir()
	ledgerPath := filepath.Join(dir, "ledger.jsonl")
	p := filepath.Join(dir, "exposures.jsonl")
	rows := []contract.ExposureEvent{{
		ExposureID: "x1", DecisionID: "nope", ItemID: "i1", Position: 1,
		OccurredAt: time.Date(2026, 8, 20, 12, 0, 0, 0, time.UTC), Status: contract.ExposureShown,
	}}
	if err := contract.WriteJSONL(p, rows); err != nil {
		t.Fatal(err)
	}
	if err := cmdAppend([]string{"-ledger", ledgerPath, "-in", p}, "exposure"); err == nil ||
		!strings.Contains(err.Error(), "unknown decision") {
		t.Fatalf("want orphan failure, got %v", err)
	}
}

func TestControlReleaseBlockedGateFails(t *testing.T) {
	dir := t.TempDir()
	model := filepath.Join(dir, "m.leaves.json")
	if err := os.WriteFile(model, []byte(`{"m":1}`), 0o644); err != nil {
		t.Fatal(err)
	}
	evalPath := filepath.Join(dir, "evaluation.json")
	writeJSON(t, evalPath, evalReportFile{
		Metrics: map[string]float64{"ndcg@10": 0.1},
		Gates: []contract.GateResult{
			{Layer: eval.LayerData, Name: "data_leakage_rate", Status: contract.StatusOK},
			{Layer: eval.LayerCandidateRank, Name: "recall_at_k", Status: contract.StatusBlock, Reason: "too low"},
			{Layer: eval.LayerDeal, Name: "deck_fill_rate", Status: contract.StatusOK},
		},
	})
	statePath := filepath.Join(dir, "release_state.json")
	err := cmdRelease([]string{
		"-state", statePath, "-action", "candidate",
		"-release-id", "rel-1", "-evaluation", evalPath, "-model", model,
		"-run-id", "r1", "-snapshot-id", "s1", "-policy-version", "p1",
		"-at", "2026-08-20T12:00:00Z",
	})
	if err == nil || !strings.Contains(err.Error(), "blocked") {
		t.Fatalf("want blocked gate failure, got %v", err)
	}
	if _, statErr := os.Stat(statePath); !os.IsNotExist(statErr) {
		t.Fatal("blocked candidate must not write state file")
	}
}

func TestControlUsageExitCode(t *testing.T) {
	if err := cmdSnapshot([]string{"-workspace", "", "-snapshot-id", "", "-purpose", ""}); exitCode(err) != 1 {
		t.Fatalf("usage error must exit 1, got %d", exitCode(err))
	}
	if err := cmdSnapshot([]string{
		"-workspace", "nope", "-snapshot-id", "x", "-purpose", "train",
		"-time-start", "2026-08-01T00:00:00Z", "-time-end", "2026-08-20T00:00:00Z",
	}); exitCode(err) != 2 {
		t.Fatalf("missing workspace artifacts must exit 2, got %d", exitCode(err))
	}
	// 契约要求时间范围：缺失 → 用法错误（exit 1），而非运行时校验失败
	if err := cmdSnapshot([]string{"-workspace", t.TempDir(), "-snapshot-id", "x", "-purpose", "train"}); exitCode(err) != 1 ||
		!strings.Contains(err.Error(), "-time-start") {
		t.Fatalf("snapshot without time range must be usage error, got %v", err)
	}
}

// ── helpers ─────────────────────────────────────────────────

func readDecisions(t *testing.T, path string) []contract.DecisionEvent {
	t.Helper()
	type env struct {
		Kind string          `json:"kind"`
		Data json.RawMessage `json:"data"`
	}
	rows, err := contract.ReadJSONL[env](path)
	if err != nil {
		t.Fatal(err)
	}
	var out []contract.DecisionEvent
	for _, e := range rows {
		if e.Kind != "decision" {
			continue
		}
		var d contract.DecisionEvent
		if err := json.Unmarshal(e.Data, &d); err != nil {
			t.Fatal(err)
		}
		out = append(out, d)
	}
	return out
}

func writeJSONL[T any](t *testing.T, path string, rows []T) {
	t.Helper()
	if err := contract.WriteJSONL(path, rows); err != nil {
		t.Fatal(err)
	}
}

func mustRead(t *testing.T, path string) []byte {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	return b
}

func ptr(v float64) *float64 { return &v }

func sortedKeys2(m map[string]recsys.DealRow) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}
