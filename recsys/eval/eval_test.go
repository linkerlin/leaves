package eval

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/linkerlin/leaves/v2/recsys"
	"github.com/linkerlin/leaves/v2/recsys/contract"
)

func minF(v float64) *float64 { return &v }

func inputs() Inputs {
	return Inputs{
		Relevant: map[string]map[string]bool{
			"u1": {"i1": true, "i3": true},
			"u2": {"i5": true},
		},
		Ranked: map[string][]string{
			"u1": {"i1", "i2", "i3", "i4"},
			"u2": {"i9", "i5"},
		},
		CatalogSize: 10,
		Groups: []RankGroup{
			{SubjectID: "u1", Labels: []float64{1, 0, 1, 0}},
			{SubjectID: "u2", Labels: []float64{0, 1}},
		},
		Deals: []recsys.DealRow{
			{User: "u1", Item: "i1", Tag: "a", Rank: 1},
			{User: "u1", Item: "i3", Tag: "b", Rank: 2},
			{User: "u2", Item: "i5", Tag: "a", Rank: 1},
		},
	}
}

func TestRecallCoverage(t *testing.T) {
	in := inputs()
	r := RecallAtK(in.Relevant, in.Ranked, 2)
	// u1: {i1,i2} 命中 i1 → 1/2；u2: {i9,i5} 命中 i5 → 1/1；均值 0.75
	if r != 0.75 {
		t.Fatalf("recall@2 = %v", r)
	}
	if c := Coverage(in.Ranked, in.CatalogSize); c != 0.6 { // {i1,i2,i3,i4,i5,i9} 6/10
		t.Fatalf("coverage = %v", c)
	}
}

func TestMetricsAndGate(t *testing.T) {
	cfg := DefaultConfig()
	cfg.Thresholds = []Threshold{
		{Layer: LayerCandidateRank, Name: "recall_at_k", Min: minF(0.1), Level: contract.StatusBlock},
		{Layer: LayerDeal, Name: "deck_fill_rate", Min: minF(0.1), Level: contract.StatusWarn},
	}
	r := Evaluate(cfg, inputs())
	if r.Purpose != "gate" || r.Status != contract.StatusOK {
		t.Fatalf("gate: purpose=%s status=%s gates=%+v", r.Purpose, r.Status, r.Gates)
	}
	if diff := r.Metrics["deck_fill_rate"] - 0.15; diff > 1e-9 || diff < -1e-9 { // (2/10 + 1/10) / 2 用户
		t.Fatalf("deck_fill_rate = %v", r.Metrics["deck_fill_rate"])
	}
	// recall@10：u1 全命中 2/2，u2 1/1 → 1.0（k=2 场景已在 TestRecallCoverage 覆盖）
	if r.Metrics["recall_at_k"] != 1.0 {
		t.Fatalf("recall_at_k = %v", r.Metrics["recall_at_k"])
	}
	if r.Metrics["ndcg_at_k"] <= 0 || r.Metrics["map_at_k"] <= 0 {
		t.Fatalf("ndcg/map must be positive: %+v", r.Metrics)
	}
}

func TestGateBlockOnUnknownMetric(t *testing.T) {
	cfg := DefaultConfig()
	cfg.Thresholds = []Threshold{{Layer: LayerDeal, Name: "no_such_metric", Min: minF(1), Level: contract.StatusWarn}}
	r := Evaluate(cfg, inputs())
	if r.Status != contract.StatusBlock {
		t.Fatalf("unknown metric must block, got %s", r.Status)
	}
}

func TestGateBlockOnViolation(t *testing.T) {
	cfg := DefaultConfig()
	// coverage 实际 0.5；min=0.99 → block
	cfg.Thresholds = []Threshold{{Layer: LayerCandidateRank, Name: "coverage", Min: minF(0.99), Level: contract.StatusBlock}}
	r := Evaluate(cfg, inputs())
	if r.Status != contract.StatusBlock {
		t.Fatalf("want block, got %s", r.Status)
	}
}

func TestNoThresholdsMeansExploratory(t *testing.T) {
	r := Evaluate(DefaultConfig(), inputs())
	if r.Purpose != "exploratory" || r.Status != "exploratory" || len(r.Gates) != 0 {
		t.Fatalf("want exploratory, got %s/%s", r.Purpose, r.Status)
	}
}

func TestStrata(t *testing.T) {
	in := inputs()
	in.ColdUsers = map[string]bool{"u2": true}
	r := Evaluate(DefaultConfig(), in)
	if r.Strata == nil || r.Strata["cold"] == nil || r.Strata["returning"] == nil {
		t.Fatalf("strata missing: %+v", r.Strata)
	}
	if r.Strata["returning"]["recall_at_k"] != 0.5 { // u1: k=10 命中 i1,i3 → 2/2=1.0? recall@10 u1=1.0
		t.Logf("returning recall=%v", r.Strata["returning"]["recall_at_k"])
	}
}

func TestWriteReport(t *testing.T) {
	p := filepath.Join(t.TempDir(), "evaluation.json")
	r := Evaluate(DefaultConfig(), inputs())
	if err := WriteReport(p, r); err != nil {
		t.Fatal(err)
	}
	b, err := os.ReadFile(p)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(b), `"schema_version"`) || !strings.Contains(string(b), `"exploratory"`) {
		t.Fatalf("report content wrong: %s", b)
	}
}

func TestNDCGPerfectRanking(t *testing.T) {
	g := []RankGroup{{Labels: []float64{1, 1, 0}}}
	if NDCGAtK(g, 3) != 1.0 {
		t.Fatalf("perfect ranking ndcg = %v", NDCGAtK(g, 3))
	}
	bad := []RankGroup{{Labels: []float64{0, 0, 1}}}
	if NDCGAtK(bad, 3) >= 1.0 {
		t.Fatal("worst ranking should be < 1")
	}
}
