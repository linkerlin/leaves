package release

import (
	"strings"
	"testing"

	"github.com/linkerlin/leaves/v2/recsys/contract"
	"github.com/linkerlin/leaves/v2/recsys/eval"
)

func TestAutoApproveHappyPath(t *testing.T) {
	mp, mh := modelFile(t)
	m := NewMachine("rel-auto-1")
	if err := m.ToCandidate(evidence("rel-auto-1", mh, fullGates()), mp, utc("2026-08-20T01:00:00Z")); err != nil {
		t.Fatal(err)
	}
	pol := AutoApprovePolicy{Label: "ci-gates", RequireAllGatesPass: true}
	if err := m.AutoApprove(pol, utc("2026-08-20T01:05:00Z")); err != nil {
		t.Fatal(err)
	}
	if m.State() != StateApproved {
		t.Fatalf("state: got %s want approved", m.State())
	}
	if got := m.Evidence().ApprovedBy; got != "auto:ci-gates" {
		t.Fatalf("approved_by: got %q want auto:ci-gates", got)
	}
	hist := m.History()
	last := hist[len(hist)-1]
	if !strings.Contains(last.Reason, `policy "ci-gates"`) || !strings.Contains(last.Reason, "ok=3") {
		t.Fatalf("audit reason missing policy/gate stats: %q", last.Reason)
	}
	// 自动批准后与人工路径等价：可 ConfirmPromoted
	if err := m.ConfirmPromoted(utc("2026-08-20T01:06:00Z")); err != nil {
		t.Fatal(err)
	}
}

func warnGates() []contract.GateResult {
	return []contract.GateResult{
		{Layer: eval.LayerData, Name: "snapshot_hash", Status: contract.StatusOK},
		{Layer: eval.LayerCandidateRank, Name: "recall_at_k", Status: contract.StatusWarn, Reason: "below target"},
		{Layer: eval.LayerDeal, Name: "deck_fill_rate", Status: contract.StatusOK},
	}
}

func TestAutoApproveRejectsWarnGates(t *testing.T) {
	mp, mh := modelFile(t)
	m := NewMachine("rel-auto-2")
	if err := m.ToCandidate(evidence("rel-auto-2", mh, warnGates()), mp, utc("2026-08-20T01:00:00Z")); err != nil {
		t.Fatal(err)
	}
	// 零容忍（默认）与 RequireAllGatesPass 均拒绝
	for _, pol := range []AutoApprovePolicy{
		{Label: "zero"},
		{Label: "strict", RequireAllGatesPass: true},
	} {
		if err := m.AutoApprove(pol, utc("2026-08-20T01:05:00Z")); err == nil ||
			!strings.Contains(err.Error(), "warn gates") {
			t.Fatalf("policy %+v: want warn rejection, got %v", pol, err)
		}
	}
	if m.State() != StateCandidate {
		t.Fatalf("state drifted: %s", m.State())
	}
	// 显式放宽到 1 个 warn 才放行
	if err := m.AutoApprove(AutoApprovePolicy{Label: "tolerant", MaxWarnGates: 1}, utc("2026-08-20T01:06:00Z")); err != nil {
		t.Fatal(err)
	}
	if m.State() != StateApproved {
		t.Fatalf("state: got %s want approved", m.State())
	}
}

func TestAutoApproveRequiresLabelAndState(t *testing.T) {
	mp, mh := modelFile(t)
	m := NewMachine("rel-auto-3")
	// 非 candidate 状态拒绝
	if err := m.AutoApprove(AutoApprovePolicy{Label: "x"}, utc("2026-08-20T01:00:00Z")); err == nil {
		t.Fatal("expected state guard error")
	}
	if err := m.ToCandidate(evidence("rel-auto-3", mh, fullGates()), mp, utc("2026-08-20T01:00:00Z")); err != nil {
		t.Fatal(err)
	}
	// 空 Label 拒绝
	if err := m.AutoApprove(AutoApprovePolicy{}, utc("2026-08-20T01:05:00Z")); err == nil ||
		!strings.Contains(err.Error(), "Label") {
		t.Fatalf("want label error, got %v", err)
	}
	if m.State() != StateCandidate {
		t.Fatalf("state drifted: %s", m.State())
	}
}
