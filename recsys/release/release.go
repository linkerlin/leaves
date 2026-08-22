// Package release 候选发布状态机、证据校验与 adapter-neutral 推广/回滚请求
// （演进方案 §17.6 / RC-08、RC-09）。
//
// 边界：状态转移只接受完整 evidence；人工批准默认开启；
// last_known_good 只引用已完整记录的 ReleaseEvidence，不可隐式漂移；
// 本包不执行任何网络副作用——真实 registry/serving/CI adapter 由应用仓库实现。
package release

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/linkerlin/leaves/v2/recsys/contract"
	"github.com/linkerlin/leaves/v2/recsys/eval"
)

// State 发布状态机状态（演进方案 §17.6）。
type State string

const (
	StateExploratory       State = "exploratory"
	StateCandidate         State = "candidate"
	StateApproved          State = "approved"
	StatePromoted          State = "promoted"
	StateObserving         State = "observing"
	StateRetrainRequested  State = "retrain_requested"
	StateRollbackRequested State = "rollback_requested"
	StateRetired           State = "retired"
)

// Transition 一次状态变迁（run_status 事件）。
type Transition struct {
	From      State     `json:"from"`
	To        State     `json:"to"`
	At        time.Time `json:"at"`
	ReleaseID string    `json:"release_id"`
	Reason    string    `json:"reason"`
}

// Machine 单个 release 的状态机。
type Machine struct {
	mu            sync.Mutex
	ReleaseID     string
	state         State
	evidence      *contract.ReleaseEvidence
	lastKnownGood string // ReleaseID of last promoted+confirmed evidence
	history       []Transition
}

// NewMachine 从 exploratory 起步。
func NewMachine(releaseID string) *Machine {
	return &Machine{ReleaseID: releaseID, state: StateExploratory}
}

// State 当前状态。
func (m *Machine) State() State {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.state
}

// LastKnownGood 当前回滚锚点（release ID，空表示尚无已验证锚点）。
func (m *Machine) LastKnownGood() string {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.lastKnownGood
}

// History 状态变迁历史（run_status 事件流）。
func (m *Machine) History() []Transition {
	m.mu.Lock()
	defer m.mu.Unlock()
	return append([]Transition(nil), m.history...)
}

func (m *Machine) transit(to State, at time.Time, reason string) {
	m.history = append(m.history, Transition{
		From: m.state, To: to, At: at, ReleaseID: m.ReleaseID, Reason: reason,
	})
	m.state = to
}

func (m *Machine) requireState(want State) error {
	if m.state != want {
		return fmt.Errorf("release: %s: transition from %s requires %s", m.ReleaseID, m.state, want)
	}
	return nil
}

// ToCandidate 接受完整证据并晋级 candidate。
// 校验：evidence 合法、三层门禁齐全且无 block、模型文件 hash 与记录一致。
func (m *Machine) ToCandidate(ev contract.ReleaseEvidence, modelPath string, at time.Time) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if err := m.requireState(StateExploratory); err != nil {
		return err
	}
	if ev.ReleaseID != m.ReleaseID {
		return fmt.Errorf("release: evidence release_id %s != machine %s", ev.ReleaseID, m.ReleaseID)
	}
	if err := contract.ValidateEvidence(&ev); err != nil {
		return err
	}
	// 三层门禁必须齐全（数据/候选排序/发牌）且无 block。
	// 阈值缺失的 exploratory 结果不可能凑齐三层 → 天然阻断（RC-04）。
	layers := map[string]bool{}
	for _, g := range ev.Gates {
		if g.Status == contract.StatusBlock {
			return fmt.Errorf("release: gate %s/%s blocked: %s", g.Layer, g.Name, g.Reason)
		}
		layers[g.Layer] = true
	}
	for _, layer := range []string{eval.LayerData, eval.LayerCandidateRank, eval.LayerDeal} {
		if !layers[layer] {
			return fmt.Errorf("release: evidence missing gate layer %s (exploratory 结果不得进入 candidate)", layer)
		}
	}
	// 模型 hash 与实际文件一致（RC-02：任一 hash 不匹配失败）。
	got, err := contract.HashFile(modelPath)
	if err != nil {
		return err
	}
	if got != ev.ModelSHA256 {
		return fmt.Errorf("release: model hash mismatch: evidence %s got %s", ev.ModelSHA256, got)
	}
	m.evidence = &ev
	m.transit(StateCandidate, at, "gates passed")
	return nil
}

// Approve 人工批准（默认必经步骤）。
func (m *Machine) Approve(approver string, at time.Time) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if err := m.requireState(StateCandidate); err != nil {
		return err
	}
	if approver == "" {
		return fmt.Errorf("release: %s: manual approval requires approver id", m.ReleaseID)
	}
	m.evidence.ApprovedBy = approver
	m.transit(StateApproved, at, "approved by "+approver)
	return nil
}

// AutoApprovePolicy 自动批准策略（RAUTO）。
//
// 前提（启用方责任，文档化不可代码保证）：启用进程自身已具备签名/访问控制——
// 状态机仍只产出 desired-state 请求（adapter 执行），本包不做网络副作用。
type AutoApprovePolicy struct {
	// Label 策略名（必填）；记入 ApprovedBy="auto:<Label>" 与变迁 reason，供审计回溯。
	Label string
	// RequireAllGatesPass 要求三层门禁全部 status=ok（出现 warn 即拒绝自动批准）。
	RequireAllGatesPass bool
	// MaxWarnGates 允许的 warn 门禁数上限（RequireAllGatesPass=false 时生效；默认 0=零容忍）。
	MaxWarnGates int
}

// AutoApprove 按策略把 candidate 自动晋级 approved（RAUTO）。
// 与人工 Approve 等价留痕；拒绝路径返回可行动错误（不静默放行）。
func (m *Machine) AutoApprove(policy AutoApprovePolicy, at time.Time) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if err := m.requireState(StateCandidate); err != nil {
		return err
	}
	if strings.TrimSpace(policy.Label) == "" {
		return fmt.Errorf("release: %s: auto-approve requires policy Label", m.ReleaseID)
	}
	nOK, nWarn := 0, 0
	for _, g := range m.evidence.Gates {
		switch g.Status {
		case contract.StatusOK:
			nOK++
		case contract.StatusWarn:
			nWarn++
		}
	}
	if policy.RequireAllGatesPass && nWarn > 0 {
		return fmt.Errorf("release: %s: auto-approve rejected: %d warn gates (RequireAllGatesPass)", m.ReleaseID, nWarn)
	}
	if !policy.RequireAllGatesPass && nWarn > policy.MaxWarnGates {
		return fmt.Errorf("release: %s: auto-approve rejected: %d warn gates > MaxWarnGates %d", m.ReleaseID, nWarn, policy.MaxWarnGates)
	}
	m.evidence.ApprovedBy = "auto:" + policy.Label
	m.transit(StateApproved, at,
		fmt.Sprintf("auto-approved by policy %q (gates ok=%d warn=%d)", policy.Label, nOK, nWarn))
	return nil
}

// ConfirmPromoted 外部 adapter 确认目标版本已生效；同时锁定 last_known_good 锚点。
func (m *Machine) ConfirmPromoted(at time.Time) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if err := m.requireState(StateApproved); err != nil {
		return err
	}
	m.transit(StatePromoted, at, "external adapter confirmed")
	return nil
}

// Observe 进入观察期。
func (m *Machine) Observe(at time.Time) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if err := m.requireState(StatePromoted); err != nil {
		return err
	}
	if m.lastKnownGood == "" {
		m.lastKnownGood = m.ReleaseID // 首个 promoted 版本成为锚点
	}
	m.transit(StateObserving, at, "observing")
	return nil
}

// RequestRetrain 观察期触发再训练（新快照/漂移/退化/定时）。
func (m *Machine) RequestRetrain(reason string, at time.Time) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if err := m.requireState(StateObserving); err != nil {
		return err
	}
	m.transit(StateRetrainRequested, at, reason)
	return nil
}

// RequestRollback 观察期硬护栏失败 → 指向 last_known_good 的回滚请求。
// 锚点必须存在且不等于当前 release（不可隐式漂移）。
func (m *Machine) RequestRollback(reason string, at time.Time) (RollbackRequest, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if err := m.requireState(StateObserving); err != nil {
		return RollbackRequest{}, err
	}
	if m.lastKnownGood == "" {
		return RollbackRequest{}, fmt.Errorf("release: %s: no verified last_known_good to roll back to", m.ReleaseID)
	}
	if m.lastKnownGood == m.ReleaseID {
		return RollbackRequest{}, fmt.Errorf("release: %s: last_known_good == self; need a prior promoted evidence", m.ReleaseID)
	}
	req := RollbackRequest{From: m.ReleaseID, To: m.lastKnownGood, Reason: reason}
	m.transit(StateRollbackRequested, at, reason)
	return req, nil
}

// Retire 数据/特征/合规条件失效。
func (m *Machine) Retire(reason string, at time.Time) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.state == StateRollbackRequested || m.state == StateObserving || m.state == StatePromoted {
		m.transit(StateRetired, at, reason)
		return nil
	}
	return fmt.Errorf("release: %s: retire not allowed from %s", m.ReleaseID, m.state)
}

// SetLastKnownGood 显式设置回滚锚点（必须指向已完整记录的证据 ID，且非自身）。
// 隐式「当前账本最佳 NDCG」不允许作为锚点。
func (m *Machine) SetLastKnownGood(releaseID string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if releaseID == "" || releaseID == m.ReleaseID {
		return fmt.Errorf("release: %s: last_known_good must reference another recorded evidence", m.ReleaseID)
	}
	m.lastKnownGood = releaseID
	return nil
}

// Evidence 当前证据（nil = 尚未晋级 candidate）。
func (m *Machine) Evidence() *contract.ReleaseEvidence {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.evidence == nil {
		return nil
	}
	ev := *m.evidence
	return &ev
}

// MachineState 机器可序列化状态（CLI/Agent 持久化与重构用；release_state.json）。
type MachineState struct {
	ReleaseID     string                    `json:"release_id"`
	State         State                     `json:"state"`
	Evidence      *contract.ReleaseEvidence `json:"evidence,omitempty"`
	LastKnownGood string                    `json:"last_known_good,omitempty"`
	History       []Transition              `json:"history"`
}

// Export 导出可持久化状态。
func (m *Machine) Export() MachineState {
	m.mu.Lock()
	defer m.mu.Unlock()
	s := MachineState{
		ReleaseID:     m.ReleaseID,
		State:         m.state,
		LastKnownGood: m.lastKnownGood,
		History:       append([]Transition(nil), m.history...),
	}
	if m.evidence != nil {
		ev := *m.evidence
		s.Evidence = &ev
	}
	return s
}

// FromState 从持久化状态重构机器（轻校验：已知状态、证据齐全、历史尾部一致）。
func FromState(s MachineState) (*Machine, error) {
	if s.ReleaseID == "" {
		return nil, fmt.Errorf("release: state release_id is empty")
	}
	switch s.State {
	case StateExploratory, StateCandidate, StateApproved, StatePromoted,
		StateObserving, StateRetrainRequested, StateRollbackRequested, StateRetired:
	default:
		return nil, fmt.Errorf("release: state %q unknown", s.State)
	}
	if s.State != StateExploratory && s.Evidence == nil {
		return nil, fmt.Errorf("release: state %s requires evidence", s.State)
	}
	if s.Evidence != nil {
		if err := contract.ValidateEvidence(s.Evidence); err != nil {
			return nil, err
		}
	}
	if n := len(s.History); n > 0 && s.History[n-1].To != s.State {
		return nil, fmt.Errorf("release: history tail %s != state %s", s.History[n-1].To, s.State)
	}
	m := &Machine{
		ReleaseID:     s.ReleaseID,
		state:         s.State,
		lastKnownGood: s.LastKnownGood,
		history:       append([]Transition(nil), s.History...),
	}
	if s.Evidence != nil {
		ev := *s.Evidence
		m.evidence = &ev
	}
	return m, nil
}

// PromoteRequest / RollbackRequest：adapter-neutral desired-state 请求。
type PromoteRequest struct {
	ReleaseID    string `json:"release_id"`
	ModelVersion string `json:"model_version"`
	ModelSHA256  string `json:"model_sha256"`
}

type RollbackRequest struct {
	From   string `json:"from"`
	To     string `json:"to"`
	Reason string `json:"reason"`
}

// Adapter 外部发布适配器契约；真实 HTTP/registry/CI 实现留给应用仓库。
// 返回 nil 仅表示外部系统已接受请求（ConfirmPromoted 前置条件）。
type Adapter interface {
	Promote(ctx context.Context, req PromoteRequest) error
	Rollback(ctx context.Context, req RollbackRequest) error
}

// FakeAdapter 测试用内存 adapter（RC-09）。
type FakeAdapter struct {
	mu         sync.Mutex
	Promotions []PromoteRequest
	Rollbacks  []RollbackRequest
	// FailNext 让下一次调用失败（故障注入）。
	FailNext error
}

func (f *FakeAdapter) Promote(_ context.Context, req PromoteRequest) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.FailNext != nil {
		err := f.FailNext
		f.FailNext = nil
		return err
	}
	f.Promotions = append(f.Promotions, req)
	return nil
}

func (f *FakeAdapter) Rollback(_ context.Context, req RollbackRequest) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.FailNext != nil {
		err := f.FailNext
		f.FailNext = nil
		return err
	}
	f.Rollbacks = append(f.Rollbacks, req)
	return nil
}

// WriteRunStatus 追加写 run_status.jsonl（每次变迁一条机器可读事件）。
func WriteRunStatus(path string, ts []Transition) error {
	return contract.WriteJSONL(path, ts)
}

// ReadRunStatus 读回 run_status.jsonl。
func ReadRunStatus(path string) ([]Transition, error) {
	return contract.ReadJSONL[Transition](path)
}

// WriteEvidence 写 release_evidence.json。
func WriteEvidence(path string, ev *contract.ReleaseEvidence) error {
	b, err := json.MarshalIndent(ev, "", "  ")
	if err != nil {
		return fmt.Errorf("release: marshal evidence: %w", err)
	}
	if err := os.WriteFile(path, b, 0o644); err != nil {
		return fmt.Errorf("release: write %s: %w", path, err)
	}
	return nil
}
