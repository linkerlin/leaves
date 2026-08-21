package ledger

import (
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/linkerlin/leaves/v2/recsys"
	"github.com/linkerlin/leaves/v2/recsys/contract"
	"github.com/linkerlin/leaves/v2/recsys/deal"
)

func utc(s string) time.Time {
	t, err := time.Parse(time.RFC3339, s)
	if err != nil {
		panic(err)
	}
	return t
}

var schemaHash = strings.Repeat("b", 64)

func decision(id string, at time.Time) contract.DecisionEvent {
	return contract.DecisionEvent{
		DecisionID:        id,
		RequestID:         "req-" + id,
		OccurredAt:        at,
		ModelVersion:      "m-1",
		FeatureSchemaHash: schemaHash,
		CandidateSetID:    "cs-1",
		PolicyVersion:     "p-1",
		Items: []contract.DecisionItem{
			{ItemID: "i1", Rank: 1, Reason: contract.ReasonOK},
			{ItemID: "i2", Rank: 2, Reason: contract.ReasonOK},
		},
	}
}

func TestLedgerAppendAndReload(t *testing.T) {
	path := filepath.Join(t.TempDir(), "ledger.jsonl")
	l, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	d := decision("d1", utc("2026-08-20T10:00:00Z"))
	if err := l.AppendDecision(d); err != nil {
		t.Fatal(err)
	}
	x := contract.ExposureEvent{
		ExposureID: "x1", DecisionID: "d1", ItemID: "i1", Position: 1,
		OccurredAt: utc("2026-08-20T10:00:01Z"), Status: contract.ExposureShown,
	}
	if err := l.AppendExposure(x); err != nil {
		t.Fatal(err)
	}
	f := contract.FeedbackEvent{
		EventID: "f1", ExposureID: "x1",
		OccurredAt: utc("2026-08-20T10:01:00Z"), EventType: contract.EventClick, Value: 1,
	}
	if err := l.AppendFeedback(f); err != nil {
		t.Fatal(err)
	}

	// 重新打开：回放校验 + 视图一致。
	l2, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	if got := l2.Decisions(); len(got) != 1 || got[0].DecisionID != "d1" {
		t.Fatalf("decisions reload: %+v", got)
	}
	if got := l2.Exposures(); len(got) != 1 || got[0].ExposureID != "x1" {
		t.Fatalf("exposures reload: %+v", got)
	}
	if got := l2.Feedback(); len(got) != 1 || got[0].EventID != "f1" {
		t.Fatalf("feedback reload: %+v", got)
	}
}

func TestLedgerRejectsAssociationViolations(t *testing.T) {
	l, err := Open(filepath.Join(t.TempDir(), "ledger.jsonl"))
	if err != nil {
		t.Fatal(err)
	}
	at := utc("2026-08-20T10:00:00Z")

	// 曝光引用未知决策
	orphanExp := contract.ExposureEvent{
		ExposureID: "x0", DecisionID: "nope", ItemID: "i1", Position: 1,
		OccurredAt: at, Status: contract.ExposureShown,
	}
	if err := l.AppendExposure(orphanExp); err == nil || !strings.Contains(err.Error(), "unknown decision") {
		t.Fatalf("want unknown decision failure, got %v", err)
	}

	if err := l.AppendDecision(decision("d1", at)); err != nil {
		t.Fatal(err)
	}
	// 反向时间曝光
	reverse := contract.ExposureEvent{
		ExposureID: "x1", DecisionID: "d1", ItemID: "i1", Position: 1,
		OccurredAt: at.Add(-time.Minute), Status: contract.ExposureShown,
	}
	if err := l.AppendExposure(reverse); err == nil || !strings.Contains(err.Error(), "reverse time") {
		t.Fatalf("want reverse time failure, got %v", err)
	}
	// 曝光条目不在决策 Top-K
	notInDeck := contract.ExposureEvent{
		ExposureID: "x2", DecisionID: "d1", ItemID: "i9", Position: 1,
		OccurredAt: at, Status: contract.ExposureShown,
	}
	if err := l.AppendExposure(notInDeck); err == nil || !strings.Contains(err.Error(), "not in decision") {
		t.Fatalf("want not-in-deck failure, got %v", err)
	}
	// 反馈引用未知曝光
	orphanFb := contract.FeedbackEvent{
		EventID: "f0", ExposureID: "nope",
		OccurredAt: at, EventType: contract.EventClick,
	}
	if err := l.AppendFeedback(orphanFb); err == nil || !strings.Contains(err.Error(), "unknown exposure") {
		t.Fatalf("want orphan feedback failure, got %v", err)
	}
	// 重复决策 ID
	if err := l.AppendDecision(decision("d1", at)); err == nil || !strings.Contains(err.Error(), "duplicate") {
		t.Fatalf("want duplicate failure, got %v", err)
	}
}

func TestDecisionFromDealMapsReasonCodes(t *testing.T) {
	deck := []recsys.DealRow{
		{User: "u1", Item: "i1", Tag: "a", Rank: 1},
		{User: "u1", Item: "i2", Tag: "a", Rank: 2},
		{User: "u1", Item: "i3", Tag: "a", Rank: 3},
	}
	log := deal.LogEntry{User: "u1", InputCandidates: 10, AfterDedup: 9, AfterTagFilter: 2, Overflow: true}
	meta := DecisionMeta{
		DecisionID: "d1", RequestID: "r1", OccurredAt: utc("2026-08-20T10:00:00Z"),
		ModelVersion: "m-1", FeatureSchemaHash: schemaHash, CandidateSetID: "cs-1", PolicyVersion: "p-1",
	}
	ev, err := DecisionFromDeal("u1", deck, log, meta)
	if err != nil {
		t.Fatal(err)
	}
	// AfterTagFilter=2 → rank>=3 的补位条目为 tag_overflow
	if ev.Items[1].Reason != contract.ReasonOK || ev.Items[2].Reason != contract.ReasonTagOverflow {
		t.Fatalf("reason mapping wrong: %+v", ev.Items)
	}
	if err := contract.ValidateDecision(&ev); err != nil {
		t.Fatal(err)
	}
}

func TestLedgerNoPIIFields(t *testing.T) {
	// 契约结构仅含匿名键与版本：手工构造含 PII 的 subject 直接被拒。
	l, _ := Open(filepath.Join(t.TempDir(), "ledger.jsonl"))
	bad := decision("d1", utc("2026-08-20T10:00:00Z"))
	bad.Items[0].ItemID = "user@mail.com"
	if err := l.AppendDecision(bad); err == nil {
		t.Fatal("want PII rejection")
	}
}
