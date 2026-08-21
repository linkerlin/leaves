package replay

import (
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/linkerlin/leaves/v2/recsys/contract"
	"github.com/linkerlin/leaves/v2/recsys/ledger"
)

func utc(s string) time.Time {
	t, err := time.Parse(time.RFC3339, s)
	if err != nil {
		panic(err)
	}
	return t
}

var schemaHash = strings.Repeat("b", 64)

type fixture struct {
	l *ledger.Ledger
}

// seed 写入一条决策（subject + 两个物品），返回决策时间。
func (f *fixture) seed(t *testing.T, decisionID, subject string, at time.Time, items ...string) {
	d := contract.DecisionEvent{
		DecisionID: decisionID, RequestID: "req-" + decisionID, SubjectID: subject,
		OccurredAt: at, ModelVersion: "m-1", FeatureSchemaHash: schemaHash,
		CandidateSetID: "cs-1", PolicyVersion: "p-1",
	}
	for i, it := range items {
		d.Items = append(d.Items, contract.DecisionItem{ItemID: it, Rank: i + 1, Reason: contract.ReasonOK})
	}
	if err := f.l.AppendDecision(d); err != nil {
		t.Fatal(err)
	}
}

func (f *fixture) expose(t *testing.T, exposureID, decisionID, item string, at time.Time, status string, position int) {
	if err := f.l.AppendExposure(contract.ExposureEvent{
		ExposureID: exposureID, DecisionID: decisionID, ItemID: item, Position: position,
		OccurredAt: at, Status: status,
	}); err != nil {
		t.Fatal(err)
	}
}

func (f *fixture) feedback(t *testing.T, eventID, exposureID string, at time.Time, typ contract.EventType, v float64) {
	fb := contract.FeedbackEvent{EventID: eventID, OccurredAt: at, EventType: typ, Value: v}
	if exposureID != "" {
		fb.ExposureID = exposureID
	} else {
		fb.DecisionID = "d1" // 无曝光引用的反馈
	}
	if err := f.l.AppendFeedback(fb); err != nil {
		t.Fatal(err)
	}
}

func newFixture(t *testing.T) *fixture {
	l, err := ledger.Open(filepath.Join(t.TempDir(), "ledger.jsonl"))
	if err != nil {
		t.Fatal(err)
	}
	return &fixture{l: l}
}

func TestBuildSamplesAttributionAndNegatives(t *testing.T) {
	f := newFixture(t)
	at := utc("2026-08-20T10:00:00Z")
	f.seed(t, "d1", "u1", at, "i1", "i2")
	f.expose(t, "x1", "d1", "i1", at.Add(time.Second), contract.ExposureShown, 1)
	f.expose(t, "x2", "d1", "i2", at.Add(time.Second), contract.ExposureShown, 2) // 无反馈 → 负样本
	f.feedback(t, "f1", "x1", at.Add(time.Minute), contract.EventClick, 1)

	samples, rep, err := BuildSamples(f.l, DefaultConfig())
	if err != nil {
		t.Fatal(err)
	}
	if rep.Positives != 1 || rep.Negatives != 1 {
		t.Fatalf("report wrong: %+v", rep)
	}
	if len(samples) != 2 {
		t.Fatalf("samples: %+v", samples)
	}
	// 确定性排序：x1 与 x2 同刻 → 按 exposure_id
	if samples[0].ExposureID != "x1" || samples[0].Label != 1 || samples[0].SubjectID != "u1" {
		t.Fatalf("positive sample wrong: %+v", samples[0])
	}
	if samples[1].ExposureID != "x2" || samples[1].Label != 0 {
		t.Fatalf("negative sample wrong: %+v", samples[1])
	}
}

func TestBuildSamplesLateAndOrphan(t *testing.T) {
	f := newFixture(t)
	at := utc("2026-08-20T10:00:00Z")
	f.seed(t, "d1", "u1", at, "i1")
	f.expose(t, "x1", "d1", "i1", at.Add(time.Second), contract.ExposureShown, 1)
	// 迟到：超出 24h 归因窗
	f.feedback(t, "late", "x1", at.Add(25*time.Hour), contract.EventClick, 1)
	// 孤立：只挂 decision
	f.feedback(t, "orphan", "", at.Add(time.Minute), contract.EventClick, 1)

	samples, rep, err := BuildSamples(f.l, DefaultConfig())
	if err != nil {
		t.Fatal(err)
	}
	if rep.LateFeedback != 1 || rep.OrphanFeedback != 1 {
		t.Fatalf("report wrong: %+v", rep)
	}
	if len(rep.DiffReasons) != 2 {
		t.Fatalf("diff reasons: %+v", rep.DiffReasons)
	}
	// 迟到反馈不构成正样本：x1 变负样本
	if rep.Positives != 0 || rep.Negatives != 1 || samples[0].Label != 0 {
		t.Fatalf("late must not be positive: %+v %+v", rep, samples)
	}
}

func TestBuildSamplesSuppressedExposureExcluded(t *testing.T) {
	f := newFixture(t)
	at := utc("2026-08-20T10:00:00Z")
	f.seed(t, "d1", "u1", at, "i1")
	f.expose(t, "x1", "d1", "i1", at.Add(time.Second), contract.ExposureSuppressed, 1)
	feedbackOnSuppressed := contract.FeedbackEvent{EventID: "f1", ExposureID: "x1", OccurredAt: at.Add(time.Minute), EventType: contract.EventClick, Value: 1}
	if err := f.l.AppendFeedback(feedbackOnSuppressed); err != nil {
		t.Fatal(err)
	}
	samples, rep, err := BuildSamples(f.l, DefaultConfig())
	if err != nil {
		t.Fatal(err)
	}
	if len(samples) != 0 || rep.SuppressedFeedback != 1 {
		t.Fatalf("suppressed must not sample: %+v %+v", samples, rep)
	}
}

func TestBuildSamplesDedup(t *testing.T) {
	f := newFixture(t)
	at := utc("2026-08-20T10:00:00Z")
	f.seed(t, "d1", "u1", at, "i1")
	f.expose(t, "x1", "d1", "i1", at.Add(time.Second), contract.ExposureShown, 1)
	f.feedback(t, "f1", "x1", at.Add(time.Minute), contract.EventClick, 1)
	f.feedback(t, "f2", "x1", at.Add(2*time.Minute), contract.EventConversion, 3)

	samples, rep, err := BuildSamples(f.l, DefaultConfig())
	if err != nil {
		t.Fatal(err)
	}
	if rep.DedupedFeedback != 1 || rep.Positives != 1 {
		t.Fatalf("dedup wrong: %+v", rep)
	}
	if samples[0].Label != 3 { // 取窗内最大值
		t.Fatalf("label should be max in-window value: %+v", samples[0])
	}
}

func TestDeterministicReplay(t *testing.T) {
	build := func() []Sample {
		f := newFixture(t)
		at := utc("2026-08-20T10:00:00Z")
		f.seed(t, "d1", "u1", at, "i1", "i2")
		f.expose(t, "x2", "d1", "i2", at.Add(time.Second), contract.ExposureShown, 2)
		f.expose(t, "x1", "d1", "i1", at.Add(time.Second), contract.ExposureShown, 1)
		s, _, err := BuildSamples(f.l, DefaultConfig())
		if err != nil {
			t.Fatal(err)
		}
		return s
	}
	a, b := build(), build()
	if len(a) != len(b) {
		t.Fatal("length mismatch")
	}
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("replay not deterministic at %d: %+v vs %+v", i, a[i], b[i])
		}
	}
}

func TestNegativePolicyNone(t *testing.T) {
	f := newFixture(t)
	at := utc("2026-08-20T10:00:00Z")
	f.seed(t, "d1", "u1", at, "i1")
	f.expose(t, "x1", "d1", "i1", at.Add(time.Second), contract.ExposureShown, 1)
	cfg := DefaultConfig()
	cfg.NegativePolicy = NegativeNone
	samples, rep, err := BuildSamples(f.l, cfg)
	if err != nil {
		t.Fatal(err)
	}
	if len(samples) != 0 || rep.Negatives != 0 {
		t.Fatalf("policy none must skip negatives: %+v", rep)
	}
	if _, _, err := BuildSamples(f.l, Config{AttributionWindow: time.Hour, NegativePolicy: "bogus"}); err == nil {
		t.Fatal("want unknown policy failure")
	}
}
