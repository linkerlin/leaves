package monitor

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/linkerlin/leaves/v2/recsys"
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

func healthyLedger(t *testing.T) *ledger.Ledger {
	l, err := ledger.Open(filepath.Join(t.TempDir(), "ledger.jsonl"))
	if err != nil {
		t.Fatal(err)
	}
	at := utc("2026-08-20T10:00:00Z")
	d := contract.DecisionEvent{
		DecisionID: "d1", RequestID: "r1", SubjectID: "u1",
		OccurredAt: at, ModelVersion: "m-1", FeatureSchemaHash: schemaHash,
		CandidateSetID: "cs-1", PolicyVersion: "p-1",
		Items: []contract.DecisionItem{
			{ItemID: "i1", Rank: 1, Reason: contract.ReasonOK},
			{ItemID: "i2", Rank: 2, Reason: contract.ReasonOK},
		},
	}
	if err := l.AppendDecision(d); err != nil {
		t.Fatal(err)
	}
	for i, id := range []string{"x1", "x2"} {
		if err := l.AppendExposure(contract.ExposureEvent{
			ExposureID: id, DecisionID: "d1", ItemID: d.Items[i].ItemID, Position: i + 1,
			OccurredAt: at.Add(time.Duration(i+1) * time.Second), Status: contract.ExposureShown,
		}); err != nil {
			t.Fatal(err)
		}
	}
	if err := l.AppendFeedback(contract.FeedbackEvent{
		EventID: "f1", ExposureID: "x1", OccurredAt: at.Add(time.Minute),
		EventType: contract.EventClick, Value: 1,
	}); err != nil {
		t.Fatal(err)
	}
	return l
}

func deals() []recsys.DealRow {
	return []recsys.DealRow{
		{User: "u1", Item: "i1", Tag: "a", Rank: 1},
		{User: "u1", Item: "i2", Tag: "b", Rank: 2},
	}
}

func minF(v float64) *float64 { return &v }
func maxF(v float64) *float64 { return &v }

func TestHealthyWindowStaysOK(t *testing.T) {
	l := healthyLedger(t)
	start, end := utc("2026-08-20T09:00:00Z"), utc("2026-08-20T11:00:00Z")
	th := []Threshold{
		{Layer: "deal", Name: "deck_fill_rate", Min: minF(0.1), Level: contract.StatusBlock},
		{Layer: "deal", Name: "ctr", Min: minF(0.1), Level: contract.StatusBlock},
		{Layer: "deal", Name: "orphan_feedback_rate", Max: maxF(0.0), Level: contract.StatusBlock},
	}
	r := BuildReport(l, start, end, deals(), 10, 3, 10, th)
	if r.Overall != contract.StatusOK {
		t.Fatalf("healthy window must be ok: %+v", r.States)
	}
	if r.Metrics["ctr"] != 0.5 { // 1 click / 2 shown
		t.Fatalf("ctr = %v", r.Metrics["ctr"])
	}
	if r.Metrics["deck_fill_rate"] != 0.2 { // 2/10 单用户
		t.Fatalf("fill = %v", r.Metrics["deck_fill_rate"])
	}
}

func TestCoverageDegradationBlocks(t *testing.T) {
	l := healthyLedger(t)
	start, end := utc("2026-08-20T09:00:00Z"), utc("2026-08-20T11:00:00Z")
	th := []Threshold{{Layer: "deal", Name: "deck_fill_rate", Min: minF(0.9), Level: contract.StatusBlock}}
	r := BuildReport(l, start, end, deals(), 10, 3, 10, th)
	if r.Overall != contract.StatusBlock {
		t.Fatalf("degraded coverage must block: %+v", r.States)
	}
}

func TestIntegrityMetricMissingBlocks(t *testing.T) {
	l := healthyLedger(t)
	th := []Threshold{{Layer: "data", Name: "no_such_metric", Min: minF(1), Level: contract.StatusWarn}}
	st := Evaluate(l2m(l), th)
	if Overall(st) != contract.StatusBlock {
		t.Fatalf("unknown metric must block integrity: %+v", st)
	}
}

func l2m(l *ledger.Ledger) map[string]float64 {
	return Aggregate(l, time.Time{}.UTC(), time.Date(3000, 1, 1, 0, 0, 0, 0, time.UTC), nil, 0, 0, 0)
}

func TestNoThresholdsUnevaluated(t *testing.T) {
	l := healthyLedger(t)
	r := BuildReport(l, utc("2026-08-20T09:00:00Z"), utc("2026-08-20T11:00:00Z"), nil, 0, 0, 0, nil)
	if r.Overall != "unevaluated" || len(r.States) != 0 {
		t.Fatalf("want unevaluated, got %s", r.Overall)
	}
}

func TestWriteReport(t *testing.T) {
	l := healthyLedger(t)
	r := BuildReport(l, utc("2026-08-20T09:00:00Z"), utc("2026-08-20T11:00:00Z"), deals(), 10, 3, 10, nil)
	p := filepath.Join(t.TempDir(), "monitor_report.json")
	if err := WriteReport(p, r); err != nil {
		t.Fatal(err)
	}
	if b := readFile(t, p); !strings.Contains(b, `"metrics"`) {
		t.Fatalf("report wrong: %s", b)
	}
}

func readFile(t *testing.T, p string) string {
	t.Helper()
	b, err := os.ReadFile(p)
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}
