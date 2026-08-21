package monitor

import (
	"testing"
	"time"

	"github.com/linkerlin/leaves/v2/recsys/contract"
)

func reportWithFill(fill float64, min float64) *Report {
	th := []Threshold{{Layer: "deal", Name: "deck_fill_rate", Min: &min, Level: contract.StatusBlock}}
	m := map[string]float64{"deck_fill_rate": fill}
	return &Report{Metrics: m, States: Evaluate(m, th), Overall: ""}
}

func TestTriggerFiresOnConsecutiveBreach(t *testing.T) {
	ts, err := NewTriggerSet([]Trigger{
		{Name: "deck-hard", Metric: "deck_fill_rate", Level: contract.StatusBlock, Consecutive: 2, Action: ActionRollback},
	})
	if err != nil {
		t.Fatal(err)
	}
	t0 := time.Date(2026, 8, 20, 12, 0, 0, 0, time.UTC)
	bad := reportWithFill(0.2, 0.8)

	if f := ts.Evaluate(bad, t0); len(f) != 0 {
		t.Fatalf("first window must not fire (consecutive=2): %+v", f)
	}
	f := ts.Evaluate(bad, t0.Add(time.Hour))
	if len(f) != 1 || f[0].Action != ActionRollback || f[0].Streak != 2 {
		t.Fatalf("second consecutive window must fire rollback: %+v", f)
	}
	// 已触发重置；再越界需重新累计
	if f := ts.Evaluate(bad, t0.Add(2*time.Hour)); len(f) != 0 {
		t.Fatalf("after firing, streak resets: %+v", f)
	}
}

func TestTriggerRecoveryResetsStreak(t *testing.T) {
	ts, _ := NewTriggerSet([]Trigger{
		{Name: "deck-hard", Metric: "deck_fill_rate", Level: contract.StatusBlock, Consecutive: 2, Action: ActionRollback},
	})
	t0 := time.Date(2026, 8, 20, 12, 0, 0, 0, time.UTC)
	bad := reportWithFill(0.2, 0.8)
	good := reportWithFill(1.0, 0.8)

	ts.Evaluate(bad, t0)
	ts.Evaluate(good, t0.Add(time.Hour)) // 恢复 → 重置
	if f := ts.Evaluate(bad, t0.Add(2*time.Hour)); len(f) != 0 {
		t.Fatalf("recovery must reset streak: %+v", f)
	}
	if f := ts.Evaluate(bad, t0.Add(3*time.Hour)); len(f) != 1 {
		t.Fatalf("two fresh breaches must fire: %+v", f)
	}
}

func TestTriggerCooldownSuppressesButKeepsStreak(t *testing.T) {
	ts, _ := NewTriggerSet([]Trigger{
		{Name: "deck-hard", Metric: "deck_fill_rate", Level: contract.StatusBlock, Consecutive: 1, Action: ActionRollback, Cooldown: 2 * time.Hour},
	})
	t0 := time.Date(2026, 8, 20, 12, 0, 0, 0, time.UTC)
	bad := reportWithFill(0.2, 0.8)

	if f := ts.Evaluate(bad, t0); len(f) != 1 {
		t.Fatalf("immediate trigger: %+v", f)
	}
	if f := ts.Evaluate(bad, t0.Add(time.Hour)); len(f) != 0 {
		t.Fatalf("cooldown must suppress: %+v", f)
	}
	if f := ts.Evaluate(bad, t0.Add(2*time.Hour)); len(f) != 1 {
		t.Fatalf("cooldown elapsed, still breached → refire: %+v", f)
	}
}

func TestTriggerLevelGate(t *testing.T) {
	// 规则只认 block：warn 级别不触发。
	warnOnly := &Report{
		Metrics: map[string]float64{"deck_fill_rate": 0.85},
		States:  []State{{Name: "deck_fill_rate", Value: 0.85, Status: contract.StatusWarn, Reason: "marginal"}},
	}
	ts, _ := NewTriggerSet([]Trigger{
		{Name: "deck-hard", Metric: "deck_fill_rate", Level: contract.StatusBlock, Consecutive: 1, Action: ActionRollback},
	})
	t0 := time.Date(2026, 8, 20, 12, 0, 0, 0, time.UTC)
	if f := ts.Evaluate(warnOnly, t0); len(f) != 0 {
		t.Fatalf("warn must not satisfy block rule: %+v", f)
	}
	// 规则认 warn：block（更严重）也应满足 warn 规则。
	blockRep := reportWithFill(0.2, 0.8)
	ts2, _ := NewTriggerSet([]Trigger{
		{Name: "deck-soft", Metric: "deck_fill_rate", Level: contract.StatusWarn, Consecutive: 1, Action: ActionRetrain},
	})
	if f := ts2.Evaluate(blockRep, t0); len(f) != 1 || f[0].Action != ActionRetrain {
		t.Fatalf("block satisfies warn rule: %+v", f)
	}
}

func TestTriggerUnconfiguredMetricResets(t *testing.T) {
	// States 缺失该指标（阈值未配置）→ 不触发且重置。
	rep := &Report{Metrics: map[string]float64{}, States: nil}
	ts, _ := NewTriggerSet([]Trigger{
		{Name: "n", Metric: "ctr", Level: contract.StatusWarn, Consecutive: 1, Action: ActionRetrain},
	})
	if f := ts.Evaluate(rep, time.Time{}.UTC()); len(f) != 0 {
		t.Fatalf("missing metric must not fire: %+v", f)
	}
}

func TestNewTriggerSetValidation(t *testing.T) {
	cases := []struct {
		r    Trigger
		want string
	}{
		{Trigger{Name: "", Metric: "m"}, "name/metric"},
		{Trigger{Name: "a", Metric: "m", Level: "maybe", Action: ActionRetrain}, "level"},
		{Trigger{Name: "a", Metric: "m", Level: contract.StatusWarn, Action: "explode"}, "action"},
		{Trigger{Name: "a", Metric: "m", Level: contract.StatusWarn, Action: ActionRetrain, Cooldown: -time.Second}, "cooldown"},
	}
	for i, c := range cases {
		if _, err := NewTriggerSet([]Trigger{c.r}); err == nil {
			t.Fatalf("case %d: want %s failure", i, c.want)
		}
	}
	if _, err := NewTriggerSet([]Trigger{
		{Name: "a", Metric: "m", Level: contract.StatusWarn, Action: ActionRetrain},
		{Name: "a", Metric: "x", Level: contract.StatusWarn, Action: ActionRetrain},
	}); err == nil {
		t.Fatal("want duplicate name failure")
	}
}
