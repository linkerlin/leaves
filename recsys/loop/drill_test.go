// Package loop 承载推荐生产闭环端到端演练（演进方案 §17.8 最低演练 / RC-11）。
// 合成确定性事件驱动：candidate → promoted → observing → 退化注入 →
// rollback 指向 last_known_good → replay 只消费可归因反馈 → retrain。
package loop

import (
	"context"
	"path/filepath"
	"sort"
	"testing"
	"time"

	"github.com/linkerlin/leaves/v2/recsys"
	"github.com/linkerlin/leaves/v2/recsys/contract"
	"github.com/linkerlin/leaves/v2/recsys/deal"
	"github.com/linkerlin/leaves/v2/recsys/eval"
	"github.com/linkerlin/leaves/v2/recsys/ledger"
	"github.com/linkerlin/leaves/v2/recsys/monitor"
	"github.com/linkerlin/leaves/v2/recsys/pipeline"
	"github.com/linkerlin/leaves/v2/recsys/release"
	"github.com/linkerlin/leaves/v2/recsys/replay"
	"github.com/linkerlin/leaves/v2/recsys/split"
	"github.com/linkerlin/leaves/v2/recsys/synth"
	"github.com/linkerlin/leaves/v2/recsys/tsvio"
)

func TestAgenticRecsysLoopDrill(t *testing.T) {
	w, err := recsys.NewWorkspace(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	cfg := recsys.DefaultSmokeConfig()
	now := time.Date(2026, 8, 20, 12, 0, 0, 0, time.UTC)
	adapter := &release.FakeAdapter{}

	// ── 1. 快照 + 时间切分 ─────────────────────────────────────
	ds, err := synth.Generate(cfg)
	if err != nil {
		t.Fatal(err)
	}
	events := timedEvents(t, ds, cfg)
	tc := split.TimeConfig{
		TrainEnd:  time.Date(2026, 8, 6, 0, 0, 0, 0, time.UTC),
		ValStart:  time.Date(2026, 8, 6, 1, 0, 0, 0, time.UTC),
		TestStart: time.Date(2026, 8, 13, 0, 0, 0, 0, time.UTC),
	}
	trainEv, _, testEv, err := split.Split(events, tc)
	if err != nil {
		t.Fatal(err)
	}
	if len(trainEv) == 0 || len(testEv) == 0 {
		t.Fatalf("split produced empty windows: train=%d test=%d", len(trainEv), len(testEv))
	}
	if err := split.CheckLeakage(trainEv, tc.ValStart); err != nil {
		t.Fatal(err)
	}
	snapshot := &contract.DatasetSnapshot{
		SnapshotID:        "snap-1",
		SchemaVersion:     contract.SchemaVersion,
		CreatedAt:         now,
		Purpose:           "release",
		FeatureSchemaHash: contract.FeatureSchemaHash(ds.FeatNames, nil),
		TimeRange:         split.TimeRange(trainEv),
	}

	// ── 2. recall/rank/deal/eval 产生 candidate evidence ─────
	res, err := pipeline.RunFromDataset(w, ds, cfg)
	if err != nil {
		t.Fatal(err)
	}
	if res.DealRows == 0 || res.Eval.TestNDCG <= 0 {
		t.Fatalf("pipeline degenerate: %+v", res.Eval)
	}
	for _, p := range []string{w.SamplesTrain(), w.SamplesTest(), w.ItemsCatalog()} {
		h, err := contract.HashFile(p)
		if err != nil {
			t.Fatal(err)
		}
		rel, err := filepath.Rel(w.Root, p)
		if err != nil {
			t.Fatal(err)
		}
		snapshot.InputFiles = append(snapshot.InputFiles, contract.FileRef{Path: filepath.ToSlash(rel), SHA256: h})
	}
	if err := contract.ValidateSnapshot(snapshot); err != nil {
		t.Fatal(err)
	}
	if err := contract.VerifyFiles(snapshot, w.Root); err != nil {
		t.Fatal(err)
	}

	testSamples, err := tsvio.ReadInteractions(w.SamplesTest())
	if err != nil {
		t.Fatal(err)
	}
	scored, err := tsvio.ReadManifestJSONL(w.RankTestScored())
	if err != nil {
		t.Fatal(err)
	}
	relevant, ranked, groups := rankViews(testSamples, scored)

	dealCfg := deal.Config{DeckSize: cfg.DeckSize, MaxSameTag: cfg.MaxSameTag}
	recent := deal.RecentItems(testSamples)
	dealRows, dealLogs, err := deal.Run(scored, recent, dealCfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := deal.Validate(dealRows, recent, cfg.MaxSameTag, cfg.DeckSize); err != nil {
		t.Fatal(err)
	}

	min := 0.0
	evalCfg := eval.DefaultConfig()
	evalCfg.RecallK = 100
	evalCfg.Thresholds = []eval.Threshold{
		{Layer: eval.LayerData, Name: "data_leakage_rate", Max: &min, Level: contract.StatusBlock},
		{Layer: eval.LayerCandidateRank, Name: "recall_at_k", Min: ptr(0.5), Level: contract.StatusBlock},
		{Layer: eval.LayerCandidateRank, Name: "ndcg_at_k", Min: ptr(0.01), Level: contract.StatusBlock},
		{Layer: eval.LayerDeal, Name: "deck_fill_rate", Min: ptr(0.8), Level: contract.StatusBlock},
	}
	report := eval.Evaluate(evalCfg, eval.Inputs{
		Relevant: relevant, Ranked: ranked, CatalogSize: len(ds.Catalog),
		Groups: groups, Deals: dealRows,
		DataLeakCount: 0, DataEventCount: len(trainEv) + len(testEv),
	})
	if err := eval.WriteReport(filepath.Join(w.MetaDir(), "evaluation.json"), report); err != nil {
		t.Fatal(err)
	}
	if report.Status != contract.StatusOK {
		t.Fatalf("gate status %s: %+v", report.Status, report.Gates)
	}

	modelHash, err := contract.HashFile(w.ModelPath())
	if err != nil {
		t.Fatal(err)
	}

	// 先让 rel-0 成为已验证锚点（首个 promoted 版本）。
	m0 := release.NewMachine("rel-0")
	if err := m0.ToCandidate(gatesEvidence("rel-0", modelHash, report), w.ModelPath(), now); err != nil {
		t.Fatal(err)
	}
	_ = m0.Approve("drill-approver", now)
	_ = m0.ConfirmPromoted(now)
	if err := m0.Observe(now); err != nil {
		t.Fatal(err)
	}

	// ── 3. fake adapter 确认 promoted ────────────────────────
	m1 := release.NewMachine("rel-1")
	if err := m1.SetLastKnownGood("rel-0"); err != nil {
		t.Fatal(err)
	}
	if err := m1.ToCandidate(gatesEvidence("rel-1", modelHash, report), w.ModelPath(), now); err != nil {
		t.Fatal(err)
	}
	if err := m1.Approve("drill-approver", now); err != nil {
		t.Fatal(err)
	}
	if err := adapter.Promote(context.Background(), release.PromoteRequest{
		ReleaseID: "rel-1", ModelVersion: "m-drill", ModelSHA256: modelHash,
	}); err != nil {
		t.Fatal(err)
	}
	if err := m1.ConfirmPromoted(now); err != nil {
		t.Fatal(err)
	}
	if err := m1.Observe(now); err != nil {
		t.Fatal(err)
	}

	// ── 4. 写 decision/exposure/feedback 账本 ────────────────
	lg, err := ledger.Open(filepath.Join(w.Root, "ledger.jsonl"))
	if err != nil {
		t.Fatal(err)
	}
	deckByUser, logByUser := groupDeal(dealRows, dealLogs)
	users := sortedUserKeys(deckByUser)
	for i, u := range users {
		ev, err := ledger.DecisionFromDeal(u, deckByUser[u], logByUser[u], ledger.DecisionMeta{
			DecisionID: "dec-" + u, RequestID: "req-" + u, SubjectID: u,
			OccurredAt:   now.Add(time.Duration(i) * time.Minute),
			ModelVersion: "m-drill", FeatureSchemaHash: snapshot.FeatureSchemaHash,
			CandidateSetID: "cs-snap-1", PolicyVersion: "deal-v1",
		})
		if err != nil {
			t.Fatal(err)
		}
		if err := lg.AppendDecision(ev); err != nil {
			t.Fatal(err)
		}
		for _, it := range ev.Items {
			if err := lg.AppendExposure(contract.ExposureEvent{
				ExposureID: "exp-" + u + "-" + it.ItemID, DecisionID: ev.DecisionID,
				ItemID: it.ItemID, Position: it.Rank,
				OccurredAt: ev.OccurredAt.Add(time.Duration(it.Rank) * time.Second),
				Status:     contract.ExposureShown,
			}); err != nil {
				t.Fatal(err)
			}
		}
	}
	// 首用户对 deck 前两名曝光：第 1 名窗内点击（正样本），第 2 名迟到反馈。
	first := users[0]
	firstTop := deckByUser[first][0].Item
	firstSecond := deckByUser[first][1].Item
	if err := lg.AppendFeedback(contract.FeedbackEvent{
		EventID: "fb-1", ExposureID: "exp-" + first + "-" + firstTop, OccurredAt: now.Add(time.Hour),
		EventType: contract.EventClick, Value: 1,
	}); err != nil {
		t.Fatal(err)
	}

	// ── 5. 健康窗口保持 observing ────────────────────────────
	winStart, winEnd := now, now.Add(2*time.Hour)
	monTh := []monitor.Threshold{
		{Layer: "deal", Name: "deck_fill_rate", Min: ptr(0.8), Level: contract.StatusBlock},
		{Layer: "data", Name: "orphan_feedback_rate", Max: ptr(0), Level: contract.StatusBlock},
	}
	healthy := monitor.BuildReport(lg, winStart, winEnd, dealRows, cfg.DeckSize, cfg.MaxSameTag, len(ds.Catalog), monTh)
	if healthy.Overall != contract.StatusOK {
		t.Fatalf("healthy window must stay ok: %+v", healthy.States)
	}
	if m1.State() != release.StateObserving {
		t.Fatalf("state drifted: %s", m1.State())
	}

	// ── 6. 注入 coverage 退化（deck 只剩 1 条/用户）──────────
	degraded := []recsys.DealRow{}
	for _, u := range users {
		degraded = append(degraded, deckByUser[u][0])
	}
	degradedReport := monitor.BuildReport(lg, winStart, winEnd, degraded, cfg.DeckSize, cfg.MaxSameTag, len(ds.Catalog), monTh)
	if degradedReport.Overall != contract.StatusBlock {
		t.Fatalf("degraded coverage must block: %+v", degradedReport.States)
	}

	// ── 7. 触发器（可配置规则 + 冷却期）产出回滚请求 ────────
	// 触发器是配置驱动的规则，不是 Agent 临场猜测（§17.6）。
	triggers, err := monitor.NewTriggerSet([]monitor.Trigger{{
		Name: "deck-fill-hard", Metric: "deck_fill_rate", Level: contract.StatusBlock,
		Consecutive: 1, Action: monitor.ActionRollback, Cooldown: time.Hour,
	}})
	if err != nil {
		t.Fatal(err)
	}
	if fired := triggers.Evaluate(healthy, now.Add(2*time.Hour)); len(fired) != 0 {
		t.Fatalf("healthy window must not fire: %+v", fired)
	}
	fired := triggers.Evaluate(degradedReport, now.Add(3*time.Hour))
	if len(fired) != 1 || fired[0].Action != monitor.ActionRollback {
		t.Fatalf("degraded window must fire rollback: %+v", fired)
	}
	if firedAgain := triggers.Evaluate(degradedReport, now.Add(3*time.Hour+30*time.Minute)); len(firedAgain) != 0 {
		t.Fatalf("cooldown must suppress refire: %+v", firedAgain)
	}
	rbReq, err := m1.RequestRollback(fired[0].Reason, now.Add(3*time.Hour))
	if err != nil {
		t.Fatal(err)
	}
	if rbReq.To != "rel-0" {
		t.Fatalf("rollback must target last_known_good rel-0, got %+v", rbReq)
	}
	if err := adapter.Rollback(context.Background(), rbReq); err != nil {
		t.Fatal(err)
	}
	if m1.State() != release.StateRollbackRequested {
		t.Fatalf("state: %s", m1.State())
	}

	// ── 8. replay 只消费可归因反馈 → 下一轮 retrain 快照 ────
	// 迟到反馈（超 24h 窗）与孤立反馈（只挂 decision）不得进入训练。
	if err := lg.AppendFeedback(contract.FeedbackEvent{
		EventID: "fb-late", ExposureID: "exp-" + first + "-" + firstSecond,
		OccurredAt: now.Add(48 * time.Hour), EventType: contract.EventConversion, Value: 5,
	}); err != nil {
		t.Fatal(err)
	}
	if err := lg.AppendFeedback(contract.FeedbackEvent{
		EventID: "fb-orphan", DecisionID: "dec-" + users[len(users)-1],
		OccurredAt: now.Add(time.Hour), EventType: contract.EventClick, Value: 1,
	}); err != nil {
		t.Fatal(err)
	}
	samples, repReport, err := replay.BuildSamples(lg, replay.DefaultConfig())
	if err != nil {
		t.Fatal(err)
	}
	if repReport.LateFeedback != 1 || repReport.OrphanFeedback != 1 {
		t.Fatalf("replay must exclude late/orphan: %+v", repReport)
	}
	if repReport.Positives != 1 {
		t.Fatalf("exactly one attributable positive expected: %+v", repReport)
	}
	if err := replay.WriteReport(filepath.Join(w.MetaDir(), "replay_report.json"), repReport); err != nil {
		t.Fatal(err)
	}

	// 下一轮快照：可归因样本成为新训练事件。
	nextEvents := make([]contract.InteractionEvent, 0, len(samples))
	for i, s := range samples {
		nextEvents = append(nextEvents, contract.InteractionEvent{
			EventID: "next-" + s.ExposureID, OccurredAt: s.ExposureAt.Add(time.Duration(i) * time.Second),
			SubjectID: s.SubjectID, ItemID: s.ItemID,
			EventType: contract.EventClick, Value: s.Label, Source: "replay-snap-2",
		})
	}
	if err := contract.ValidateInteractions(nextEvents); err != nil {
		t.Fatal(err)
	}
	snap2 := &contract.DatasetSnapshot{
		SnapshotID: "snap-2", SchemaVersion: contract.SchemaVersion, CreatedAt: now.Add(72 * time.Hour),
		Purpose: "train", FeatureSchemaHash: snapshot.FeatureSchemaHash,
		TimeRange: split.TimeRange(nextEvents),
	}
	if err := contract.ValidateSnapshot(snap2); err != nil {
		t.Fatal(err)
	}
	if err := m0.RequestRetrain("replay snapshot snap-2 ready", now.Add(73*time.Hour)); err != nil {
		t.Fatal(err)
	}
	if m0.State() != release.StateRetrainRequested {
		t.Fatalf("m0 state: %s", m0.State())
	}

	// run_status 落盘可回放。
	if err := release.WriteRunStatus(filepath.Join(w.MetaDir(), "run_status.jsonl"), m1.History()); err != nil {
		t.Fatal(err)
	}
	status, err := release.ReadRunStatus(filepath.Join(w.MetaDir(), "run_status.jsonl"))
	if err != nil || len(status) == 0 {
		t.Fatalf("run_status round trip: %v %+v", err, status)
	}
}

// timedEvents 给合成交互分配确定性时间：每用户事件均匀铺满 14 天，
// 保证 train/val/test 三个窗口都有事件。
func timedEvents(t *testing.T, ds synth.Dataset, cfg recsys.SmokeConfig) []contract.InteractionEvent {
	t.Helper()
	byUser := map[string][]recsys.Interaction{}
	var userOrder []string
	for _, r := range ds.Raw {
		if _, ok := byUser[r.User]; !ok {
			userOrder = append(userOrder, r.User)
		}
		byUser[r.User] = append(byUser[r.User], r)
	}
	sort.Strings(userOrder)
	start := time.Date(2026, 8, 1, 0, 0, 0, 0, time.UTC)
	span := 14 * 24 * time.Hour
	var out []contract.InteractionEvent
	n := 0
	for _, u := range userOrder {
		rows := byUser[u]
		for i, r := range rows {
			at := start.Add(time.Duration(float64(span) * float64(i) / float64(len(rows))))
			typ := contract.EventClick
			if i%2 == 1 {
				typ = contract.EventRating
			}
			out = append(out, contract.InteractionEvent{
				EventID: "ev-" + u + "-" + r.Item, OccurredAt: at,
				SubjectID: u, ItemID: r.Item, EventType: typ, Value: r.Score, Source: "drill",
			})
			n++
		}
	}
	if err := contract.ValidateInteractions(out); err != nil {
		t.Fatal(err)
	}
	return out
}

// rankViews 构造 eval 输入：relevant（test 正样本）、ranked（按 margin 降序）、
// Groups（标签按预测分数排序后的命中序列）。
func rankViews(testSamples []recsys.Interaction, scored []recsys.ManifestRow) (
	relevant map[string]map[string]bool, ranked map[string][]string, groups []eval.RankGroup,
) {
	relevant = map[string]map[string]bool{}
	for _, s := range testSamples {
		if relevant[s.User] == nil {
			relevant[s.User] = map[string]bool{}
		}
		relevant[s.User][s.Item] = true
	}
	byUser := map[string][]recsys.ManifestRow{}
	for _, r := range scored {
		byUser[r.User] = append(byUser[r.User], r)
	}
	ranked = map[string][]string{}
	for u, rows := range byUser {
		sorted := append([]recsys.ManifestRow(nil), rows...)
		sort.Slice(sorted, func(i, j int) bool { return sorted[i].Score > sorted[j].Score })
		list := make([]string, len(sorted))
		labels := make([]float64, len(sorted))
		for i, r := range sorted {
			list[i] = r.Item
			if relevant[u][r.Item] {
				labels[i] = 1
			}
		}
		ranked[u] = list
		groups = append(groups, eval.RankGroup{SubjectID: u, Labels: labels})
	}
	return relevant, ranked, groups
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

func sortedUserKeys(m map[string][]recsys.DealRow) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

func gatesEvidence(releaseID, modelHash string, r *eval.Report) contract.ReleaseEvidence {
	return contract.ReleaseEvidence{
		ReleaseID:      releaseID,
		ModelSHA256:    modelHash,
		RunID:          "run-" + releaseID,
		SnapshotID:     "snap-1",
		PolicyVersion:  "deal-v1",
		OfflineMetrics: r.Metrics,
		Gates:          r.Gates,
		CreatedAt:      time.Date(2026, 8, 20, 12, 0, 0, 0, time.UTC),
	}
}

func ptr(v float64) *float64 { return &v }
