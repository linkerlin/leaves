package synth

import (
	"testing"
	"time"

	"github.com/linkerlin/leaves/v2/recsys"
)

func TestGenerateDeterministic(t *testing.T) {
	cfg := recsys.SmokeConfig{
		Seed: 42, TrainUsers: 4, TestUsers: 2,
		RecallSize: 8, NumItems: 20, MinEvents: 3,
	}
	a, err := Generate(cfg)
	if err != nil {
		t.Fatal(err)
	}
	b, err := Generate(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if len(a.TrainUsers) != 4 || len(a.TestUsers) != 2 {
		t.Fatalf("users train=%d test=%d", len(a.TrainUsers), len(a.TestUsers))
	}
	if len(a.Catalog) != 20 {
		t.Fatalf("catalog=%d", len(a.Catalog))
	}
	if len(a.Raw) == 0 {
		t.Fatal("no interactions")
	}
	if len(a.Raw) != len(b.Raw) {
		t.Fatalf("len raw %d vs %d", len(a.Raw), len(b.Raw))
	}
	if a.Raw[0].User != b.Raw[0].User || a.Raw[0].Item != b.Raw[0].Item {
		t.Fatalf("seed not deterministic: %+v vs %+v", a.Raw[0], b.Raw[0])
	}
	var prev time.Time
	for i, r := range a.Raw {
		if r.Time.IsZero() {
			t.Fatalf("row %d missing time", i)
		}
		if !r.Time.Equal(r.Time.UTC()) {
			t.Fatalf("row %d not UTC: %v", i, r.Time)
		}
		if i > 0 && !r.Time.After(prev) {
			t.Fatalf("times not strictly increasing at %d", i)
		}
		prev = r.Time
	}
}

func TestGenerateRejectsBadConfig(t *testing.T) {
	_, err := Generate(recsys.SmokeConfig{TrainUsers: 0, TestUsers: 1, NumItems: 10, RecallSize: 5})
	if err == nil {
		t.Fatal("expected error for zero users")
	}
	_, err = Generate(recsys.SmokeConfig{TrainUsers: 1, TestUsers: 1, NumItems: 3, RecallSize: 10})
	if err == nil {
		t.Fatal("expected error for NumItems < RecallSize")
	}
}
