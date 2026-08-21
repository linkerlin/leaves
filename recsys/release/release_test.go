package release

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/linkerlin/leaves/v2/recsys/contract"
	"github.com/linkerlin/leaves/v2/recsys/eval"
)

func utc(s string) time.Time {
	t, err := time.Parse(time.RFC3339, s)
	if err != nil {
		panic(err)
	}
	return t
}

func modelFile(t *testing.T) (string, string) {
	t.Helper()
	p := filepath.Join(t.TempDir(), "model.leaves.json")
	if err := os.WriteFile(p, []byte(`{"model":1}`), 0o644); err != nil {
		t.Fatal(err)
	}
	h, err := contract.HashFile(p)
	if err != nil {
		t.Fatal(err)
	}
	return p, h
}

func fullGates() []contract.GateResult {
	return []contract.GateResult{
		{Layer: eval.LayerData, Name: "snapshot_hash", Status: contract.StatusOK},
		{Layer: eval.LayerCandidateRank, Name: "recall_at_k", Status: contract.StatusOK},
		{Layer: eval.LayerDeal, Name: "deck_fill_rate", Status: contract.StatusOK},
	}
}

func evidence(releaseID, modelHash string, gates []contract.GateResult) contract.ReleaseEvidence {
	return contract.ReleaseEvidence{
		ReleaseID:      releaseID,
		ModelSHA256:    modelHash,
		RunID:          "run-" + releaseID,
		SnapshotID:     "snap-1",
		PolicyVersion:  "p-1",
		OfflineMetrics: map[string]float64{"ndcg@10": 0.6},
		Gates:          gates,
		CreatedAt:      utc("2026-08-20T00:00:00Z"),
	}
}

func TestHappyPathToObserving(t *testing.T) {
	mp, mh := modelFile(t)
	m := NewMachine("rel-1")
	if err := m.ToCandidate(evidence("rel-1", mh, fullGates()), mp, utc("2026-08-20T01:00:00Z")); err != nil {
		t.Fatal(err)
	}
	if err := m.Approve("human-1", utc("2026-08-20T02:00:00Z")); err != nil {
		t.Fatal(err)
	}
	adapter := &FakeAdapter{}
	if err := adapter.Promote(context.Background(), PromoteRequest{ReleaseID: "rel-1", ModelVersion: "m-1", ModelSHA256: mh}); err != nil {
		t.Fatal(err)
	}
	if err := m.ConfirmPromoted(utc("2026-08-20T03:00:00Z")); err != nil {
		t.Fatal(err)
	}
	if err := m.Observe(utc("2026-08-20T04:00:00Z")); err != nil {
		t.Fatal(err)
	}
	if m.State() != StateObserving || m.LastKnownGood() != "rel-1" {
		t.Fatalf("state=%s lkg=%s", m.State(), m.LastKnownGood())
	}
	if len(m.History()) != 4 {
		t.Fatalf("history: %+v", m.History())
	}
}

func TestCandidateRejectsIncompleteEvidence(t *testing.T) {
	mp, mh := modelFile(t)
	m := NewMachine("rel-1")

	// 缺层：只有 deal 门禁 → 拒绝
	partial := []contract.GateResult{{Layer: eval.LayerDeal, Name: "deck_fill_rate", Status: contract.StatusOK}}
	if err := m.ToCandidate(evidence("rel-1", mh, partial), mp, utc("2026-08-20T01:00:00Z")); err == nil ||
		!strings.Contains(err.Error(), "missing gate layer") {
		t.Fatalf("want missing layer failure, got %v", err)
	}
	// block 门禁 → 拒绝
	blocked := fullGates()
	blocked[1].Status = contract.StatusBlock
	if err := m.ToCandidate(evidence("rel-1", mh, blocked), mp, utc("2026-08-20T01:00:00Z")); err == nil {
		t.Fatal("want blocked gate to fail")
	}
	// 模型 hash 不符 → 拒绝
	if err := m.ToCandidate(evidence("rel-1", strings.Repeat("d", 64), fullGates()), mp, utc("2026-08-20T01:00:00Z")); err == nil ||
		!strings.Contains(err.Error(), "model hash mismatch") {
		t.Fatalf("want hash mismatch, got %v", err)
	}
	// evidence ID 不符 → 拒绝
	if err := m.ToCandidate(evidence("rel-2", mh, fullGates()), mp, utc("2026-08-20T01:00:00Z")); err == nil {
		t.Fatal("want release id mismatch to fail")
	}
	if m.State() != StateExploratory {
		t.Fatal("must stay exploratory on rejected evidence")
	}
}

func TestApprovalIsRequiredAndManual(t *testing.T) {
	mp, mh := modelFile(t)
	m := NewMachine("rel-1")
	if err := m.ToCandidate(evidence("rel-1", mh, fullGates()), mp, utc("2026-08-20T01:00:00Z")); err != nil {
		t.Fatal(err)
	}
	if err := m.Approve("", utc("2026-08-20T02:00:00Z")); err == nil {
		t.Fatal("want empty approver to fail")
	}
	// 未批准直接确认 promoted → 拒绝
	if err := m.ConfirmPromoted(utc("2026-08-20T02:30:00Z")); err == nil {
		t.Fatal("want promotion without approval to fail")
	}
}

func TestRollbackPointsToLastKnownGood(t *testing.T) {
	mp, mh := modelFile(t)
	m := NewMachine("rel-2")
	// 显式锚点：上一个 promoted 版本
	if err := m.SetLastKnownGood("rel-1"); err != nil {
		t.Fatal(err)
	}
	if err := m.ToCandidate(evidence("rel-2", mh, fullGates()), mp, utc("2026-08-20T01:00:00Z")); err != nil {
		t.Fatal(err)
	}
	_ = m.Approve("human-1", utc("2026-08-20T02:00:00Z"))
	_ = m.ConfirmPromoted(utc("2026-08-20T03:00:00Z"))
	if err := m.Observe(utc("2026-08-20T04:00:00Z")); err != nil {
		t.Fatal(err)
	}
	// 观察期健康：锚点不变
	if m.LastKnownGood() != "rel-1" {
		t.Fatalf("lkg drifted to %s", m.LastKnownGood())
	}
	req, err := m.RequestRollback("coverage degraded", utc("2026-08-20T05:00:00Z"))
	if err != nil {
		t.Fatal(err)
	}
	if req.To != "rel-1" || req.From != "rel-2" {
		t.Fatalf("rollback target wrong: %+v", req)
	}
	adapter := &FakeAdapter{}
	if err := adapter.Rollback(context.Background(), req); err != nil {
		t.Fatal(err)
	}
	if len(adapter.Rollbacks) != 1 || adapter.Rollbacks[0].To != "rel-1" {
		t.Fatalf("adapter rollbacks: %+v", adapter.Rollbacks)
	}
	if m.State() != StateRollbackRequested {
		t.Fatal("state must be rollback_requested")
	}
}

func TestRollbackWithoutAnchorFails(t *testing.T) {
	mp, mh := modelFile(t)
	m := NewMachine("rel-1")
	_ = m.ToCandidate(evidence("rel-1", mh, fullGates()), mp, utc("2026-08-20T01:00:00Z"))
	_ = m.Approve("h", utc("2026-08-20T02:00:00Z"))
	_ = m.ConfirmPromoted(utc("2026-08-20T03:00:00Z"))
	if err := m.Observe(utc("2026-08-20T04:00:00Z")); err != nil {
		t.Fatal(err)
	}
	// 唯一 promoted 版本即自身 → 无可回滚锚点
	if _, err := m.RequestRollback("x", utc("2026-08-20T05:00:00Z")); err == nil ||
		!strings.Contains(err.Error(), "prior promoted") {
		t.Fatalf("want no-anchor failure, got %v", err)
	}
}

func TestRetrainRequest(t *testing.T) {
	mp, mh := modelFile(t)
	m := NewMachine("rel-1")
	_ = m.ToCandidate(evidence("rel-1", mh, fullGates()), mp, utc("2026-08-20T01:00:00Z"))
	if err := m.RequestRetrain("drift", utc("2026-08-20T02:00:00Z")); err == nil {
		t.Fatal("retrain only from observing")
	}
	_ = m.Approve("h", utc("2026-08-20T02:00:00Z"))
	_ = m.ConfirmPromoted(utc("2026-08-20T03:00:00Z"))
	_ = m.Observe(utc("2026-08-20T04:00:00Z"))
	if err := m.RequestRetrain("drift", utc("2026-08-20T05:00:00Z")); err != nil {
		t.Fatal(err)
	}
	if m.State() != StateRetrainRequested {
		t.Fatal("state must be retrain_requested")
	}
}

func TestRunStatusRoundTrip(t *testing.T) {
	p := filepath.Join(t.TempDir(), "run_status.jsonl")
	ts := []Transition{
		{From: StateExploratory, To: StateCandidate, At: utc("2026-08-20T01:00:00Z"), ReleaseID: "r", Reason: "gates passed"},
	}
	if err := WriteRunStatus(p, ts); err != nil {
		t.Fatal(err)
	}
	got, err := ReadRunStatus(p)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 || got[0].To != StateCandidate {
		t.Fatalf("round trip: %+v", got)
	}
}

func TestFakeAdapterFailureInjection(t *testing.T) {
	f := &FakeAdapter{FailNext: os.ErrDeadlineExceeded}
	if err := f.Promote(context.Background(), PromoteRequest{}); err == nil {
		t.Fatal("want injected failure")
	}
	if err := f.Promote(context.Background(), PromoteRequest{ReleaseID: "r"}); err != nil {
		t.Fatal(err)
	}
	if len(f.Promotions) != 1 {
		t.Fatal("second call must succeed")
	}
}
