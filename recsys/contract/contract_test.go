package contract

import (
	"strings"
	"testing"
	"time"
)

func utc(s string) time.Time {
	t, err := time.Parse(time.RFC3339, s)
	if err != nil {
		panic(err)
	}
	return t.UTC()
}

func validSnapshot() DatasetSnapshot {
	return DatasetSnapshot{
		SnapshotID:        "snap-1",
		SchemaVersion:     SchemaVersion,
		CreatedAt:         utc("2026-08-20T00:00:00Z"),
		Purpose:           "train",
		InputFiles:        []FileRef{{Path: "clean/samples_train.tsv", SHA256: strings.Repeat("a", 64)}},
		FeatureSchemaHash: strings.Repeat("b", 64),
		TimeRange:         TimeRange{Start: utc("2026-01-01T00:00:00Z"), End: utc("2026-06-01T00:00:00Z")},
	}
}

func TestSnapshotValid(t *testing.T) {
	s := validSnapshot()
	if err := ValidateSnapshot(&s); err != nil {
		t.Fatal(err)
	}
}

func TestSnapshotRejectsBadHashAndVersion(t *testing.T) {
	s := validSnapshot()
	s.FeatureSchemaHash = "nothex"
	if err := ValidateSnapshot(&s); err == nil {
		t.Fatal("want invalid feature_schema_hash to fail")
	}
	s = validSnapshot()
	s.SchemaVersion = 99
	if err := ValidateSnapshot(&s); err == nil {
		t.Fatal("want schema_version mismatch to fail")
	}
	s = validSnapshot()
	off, err := time.Parse(time.RFC3339, "2026-08-20T08:00:00+08:00") // 非 UTC 位置
	if err != nil {
		t.Fatal(err)
	}
	s.CreatedAt = off
	if err := ValidateSnapshot(&s); err == nil {
		t.Fatal("want non-UTC time to fail")
	}
	s = validSnapshot()
	s.TimeRange.End = s.TimeRange.Start.Add(-time.Hour)
	if err := ValidateSnapshot(&s); err == nil {
		t.Fatal("want reversed time range to fail")
	}
}

func TestSnapshotLegacyAllowsMissingFeatureHash(t *testing.T) {
	s := validSnapshot()
	s.FeatureSchemaHash = ""
	s.LegacySnapshot = true
	if err := ValidateSnapshot(&s); err != nil {
		t.Fatal(err)
	}
}

func TestInteractionsDuplicateIDFails(t *testing.T) {
	ev := []InteractionEvent{
		{EventID: "e1", OccurredAt: utc("2026-01-01T00:00:00Z"), SubjectID: "u1", ItemID: "i1", EventType: EventClick},
		{EventID: "e1", OccurredAt: utc("2026-01-02T00:00:00Z"), SubjectID: "u2", ItemID: "i2", EventType: EventClick},
	}
	if err := ValidateInteractions(ev); err == nil || !strings.Contains(err.Error(), "duplicate") {
		t.Fatalf("want duplicate event_id failure, got %v", err)
	}
}

func TestInteractionsUnknownTypeFails(t *testing.T) {
	ev := []InteractionEvent{
		{EventID: "e1", OccurredAt: utc("2026-01-01T00:00:00Z"), SubjectID: "u1", ItemID: "i1", EventType: EventType("view")},
	}
	if err := ValidateInteractions(ev); err == nil || !strings.Contains(err.Error(), "unknown event_type") {
		t.Fatalf("want unknown event_type failure, got %v", err)
	}
}

func TestInteractionsPIITripwire(t *testing.T) {
	ev := []InteractionEvent{
		{EventID: "e1", OccurredAt: utc("2026-01-01T00:00:00Z"), SubjectID: "user@mail.com", ItemID: "i1", EventType: EventClick},
	}
	if err := ValidateInteractions(ev); err == nil || !strings.Contains(err.Error(), "anonymized") {
		t.Fatalf("want PII tripwire failure, got %v", err)
	}
}

func decision() DecisionEvent {
	return DecisionEvent{
		DecisionID:        "d1",
		RequestID:         "r1",
		OccurredAt:        utc("2026-08-20T10:00:00Z"),
		ModelVersion:      "m-1",
		FeatureSchemaHash: strings.Repeat("b", 64),
		CandidateSetID:    "cs-1",
		PolicyVersion:     "p-1",
		Items: []DecisionItem{
			{ItemID: "i1", Rank: 1, Reason: ReasonOK},
			{ItemID: "i2", Rank: 2, Reason: ReasonTagOverflow},
		},
	}
}

func TestDecisionReasonCodeRequired(t *testing.T) {
	d := decision()
	d.Items[0].Reason = ""
	if err := ValidateDecision(&d); err == nil || !strings.Contains(err.Error(), "reason") {
		t.Fatalf("want unknown reason failure, got %v", err)
	}
	d = decision()
	d.Items[1].Reason = "magic_code"
	if err := ValidateDecision(&d); err == nil {
		t.Fatal("want unknown reason code to fail")
	}
}

func TestDecisionDuplicateRankFails(t *testing.T) {
	d := decision()
	d.Items[1].Rank = 1
	if err := ValidateDecision(&d); err == nil || !strings.Contains(err.Error(), "rank") {
		t.Fatalf("want duplicate rank failure, got %v", err)
	}
}

func TestExposureReverseTimeFails(t *testing.T) {
	d := decision()
	e := ExposureEvent{
		ExposureID: "x1", DecisionID: "d1", ItemID: "i1", Position: 1,
		OccurredAt: d.OccurredAt.Add(-time.Minute), Status: ExposureShown,
	}
	if err := ValidateExposure(&e, &d); err == nil || !strings.Contains(err.Error(), "reverse time") {
		t.Fatalf("want reverse time failure, got %v", err)
	}
}

func TestExposureUnknownDecisionFails(t *testing.T) {
	d := decision()
	e := ExposureEvent{
		ExposureID: "x1", DecisionID: "dX", ItemID: "i1", Position: 1,
		OccurredAt: d.OccurredAt, Status: ExposureShown,
	}
	if err := ValidateExposure(&e, &d); err == nil {
		t.Fatal("want decision mismatch to fail")
	}
	if err := ValidateExposure(&e, nil); err == nil {
		t.Fatal("want unknown decision to fail")
	}
}

func TestExposurePositionMustMatchRank(t *testing.T) {
	d := decision()
	e := ExposureEvent{
		ExposureID: "x1", DecisionID: "d1", ItemID: "i1", Position: 2,
		OccurredAt: d.OccurredAt, Status: ExposureShown,
	}
	if err := ValidateExposure(&e, &d); err == nil || !strings.Contains(err.Error(), "position") {
		t.Fatalf("want position mismatch failure, got %v", err)
	}
}

func TestFeedbackAssociation(t *testing.T) {
	d := decision()
	x := ExposureEvent{
		ExposureID: "x1", DecisionID: "d1", ItemID: "i1", Position: 1,
		OccurredAt: d.OccurredAt.Add(time.Second), Status: ExposureShown,
	}
	ok := FeedbackEvent{EventID: "f1", ExposureID: "x1", OccurredAt: x.OccurredAt.Add(time.Second), EventType: EventClick}
	if err := ValidateFeedback(&ok, &x); err != nil {
		t.Fatal(err)
	}
	bad := ok
	bad.OccurredAt = x.OccurredAt.Add(-time.Second)
	if err := ValidateFeedback(&bad, &x); err == nil || !strings.Contains(err.Error(), "reverse time") {
		t.Fatalf("want reverse time failure, got %v", err)
	}
	orphan := FeedbackEvent{EventID: "f2", ExposureID: "nope", OccurredAt: ok.OccurredAt, EventType: EventClick}
	if err := ValidateFeedback(&orphan, nil); err == nil || !strings.Contains(err.Error(), "unknown exposure") {
		t.Fatalf("want orphan failure, got %v", err)
	}
	notFeedback := FeedbackEvent{EventID: "f3", DecisionID: "d1", OccurredAt: ok.OccurredAt, EventType: EventImpression}
	if err := ValidateFeedback(&notFeedback, nil); err == nil || !strings.Contains(err.Error(), "feedback subset") {
		t.Fatalf("want impression rejected as feedback, got %v", err)
	}
	noRef := FeedbackEvent{EventID: "f4", OccurredAt: ok.OccurredAt, EventType: EventClick}
	if err := ValidateFeedback(&noRef, nil); err == nil {
		t.Fatal("want reference-less feedback to fail")
	}
}

func TestEvidenceValidation(t *testing.T) {
	ev := ReleaseEvidence{
		ReleaseID: "rel-1", ModelSHA256: strings.Repeat("c", 64),
		RunID: "run-1", SnapshotID: "snap-1", PolicyVersion: "p-1",
		OfflineMetrics: map[string]float64{"ndcg@10": 0.5},
		Gates:          []GateResult{{Layer: "deal", Name: "deck_fill", Status: StatusOK}},
		CreatedAt:      utc("2026-08-20T00:00:00Z"),
	}
	if err := ValidateEvidence(&ev); err != nil {
		t.Fatal(err)
	}
	ev.ModelSHA256 = "short"
	if err := ValidateEvidence(&ev); err == nil {
		t.Fatal("want bad model hash to fail")
	}
	ev.ModelSHA256 = strings.Repeat("c", 64)
	ev.Gates[0].Status = "maybe"
	if err := ValidateEvidence(&ev); err == nil {
		t.Fatal("want unknown gate status to fail")
	}
}
