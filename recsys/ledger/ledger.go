// Package ledger 决策/曝光/反馈 JSONL 账本（演进方案 §17.3 / RC-05）。
//
// DecisionEvent 是审计真源（DealRow 仅为展示行）；曝光可回链决策、
// 反馈可回链曝光；结构只含匿名键与版本，不泄漏原始 PII。
// 账本只追加（append-only）。
package ledger

import (
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"sync"
	"time"

	"github.com/linkerlin/leaves/v2/recsys"
	"github.com/linkerlin/leaves/v2/recsys/contract"
	"github.com/linkerlin/leaves/v2/recsys/deal"
)

// 行类型标签（同一 JSONL 文件内三种事件的信封）。
const (
	KindDecision = "decision"
	KindExposure = "exposure"
	KindFeedback = "feedback"
)

type envelope struct {
	Kind string          `json:"kind"`
	Data json.RawMessage `json:"data"`
}

// Ledger 内存索引 + 追加写账本。
type Ledger struct {
	mu        sync.Mutex
	path      string
	decisions map[string]*contract.DecisionEvent
	exposures map[string]*contract.ExposureEvent
	feedback  map[string]*contract.FeedbackEvent
	// 按决策聚合的曝光/反馈（audit 快路径）
	expByDecision map[string][]string
}

// Open 打开（或创建）账本文件并回放已有行。
func Open(path string) (*Ledger, error) {
	l := &Ledger{
		path:          path,
		decisions:     map[string]*contract.DecisionEvent{},
		exposures:     map[string]*contract.ExposureEvent{},
		feedback:      map[string]*contract.FeedbackEvent{},
		expByDecision: map[string][]string{},
	}
	if _, err := os.Stat(path); os.IsNotExist(err) {
		return l, nil
	}
	raws, err := contract.ReadJSONL[envelope](path)
	if err != nil {
		return nil, err
	}
	for i := range raws {
		switch raws[i].Kind {
		case KindDecision:
			var d contract.DecisionEvent
			if err := json.Unmarshal(raws[i].Data, &d); err != nil {
				return nil, fmt.Errorf("ledger: decision line %d: %w", i+1, err)
			}
			if err := l.appendDecision(&d); err != nil {
				return nil, err
			}
		case KindExposure:
			var e contract.ExposureEvent
			if err := json.Unmarshal(raws[i].Data, &e); err != nil {
				return nil, fmt.Errorf("ledger: exposure line %d: %w", i+1, err)
			}
			if err := l.appendExposure(&e); err != nil {
				return nil, err
			}
		case KindFeedback:
			var f contract.FeedbackEvent
			if err := json.Unmarshal(raws[i].Data, &f); err != nil {
				return nil, fmt.Errorf("ledger: feedback line %d: %w", i+1, err)
			}
			if err := l.appendFeedback(&f); err != nil {
				return nil, err
			}
		default:
			return nil, fmt.Errorf("ledger: unknown kind %q line %d", raws[i].Kind, i+1)
		}
	}
	return l, nil
}

// AppendDecision 校验并追加决策事件。
func (l *Ledger) AppendDecision(d contract.DecisionEvent) error {
	l.mu.Lock()
	defer l.mu.Unlock()
	if err := l.appendDecision(&d); err != nil {
		return err
	}
	return l.writeLine(envelopeOf(KindDecision, &d))
}

// AppendExposure 校验（回链决策）并追加曝光事件。
func (l *Ledger) AppendExposure(e contract.ExposureEvent) error {
	l.mu.Lock()
	defer l.mu.Unlock()
	if err := l.appendExposure(&e); err != nil {
		return err
	}
	return l.writeLine(envelopeOf(KindExposure, &e))
}

// AppendFeedback 校验（回链曝光）并追加反馈事件。
func (l *Ledger) AppendFeedback(f contract.FeedbackEvent) error {
	l.mu.Lock()
	defer l.mu.Unlock()
	if err := l.appendFeedback(&f); err != nil {
		return err
	}
	return l.writeLine(envelopeOf(KindFeedback, &f))
}

func (l *Ledger) appendDecision(d *contract.DecisionEvent) error {
	if err := contract.ValidateDecision(d); err != nil {
		return err
	}
	if _, dup := l.decisions[d.DecisionID]; dup {
		return fmt.Errorf("ledger: duplicate decision_id %s", d.DecisionID)
	}
	l.decisions[d.DecisionID] = d
	return nil
}

func (l *Ledger) appendExposure(e *contract.ExposureEvent) error {
	d, ok := l.decisions[e.DecisionID]
	if !ok {
		return fmt.Errorf("ledger: exposure %s references unknown decision %s", e.ExposureID, e.DecisionID)
	}
	if err := contract.ValidateExposure(e, d); err != nil {
		return err
	}
	if _, dup := l.exposures[e.ExposureID]; dup {
		return fmt.Errorf("ledger: duplicate exposure_id %s", e.ExposureID)
	}
	l.exposures[e.ExposureID] = e
	l.expByDecision[e.DecisionID] = append(l.expByDecision[e.DecisionID], e.ExposureID)
	return nil
}

func (l *Ledger) appendFeedback(f *contract.FeedbackEvent) error {
	var exp *contract.ExposureEvent
	if f.ExposureID != "" {
		var ok bool
		exp, ok = l.exposures[f.ExposureID]
		if !ok {
			// 关联缺失：不构成 supervised 正样本，但也拒绝入账本（可审计性要求先有曝光）。
			return fmt.Errorf("ledger: feedback %s references unknown exposure %s", f.EventID, f.ExposureID)
		}
	}
	if _, dup := l.feedback[f.EventID]; dup {
		return fmt.Errorf("ledger: duplicate feedback event_id %s", f.EventID)
	}
	if err := contract.ValidateFeedback(f, exp); err != nil {
		return err
	}
	l.feedback[f.EventID] = f
	return nil
}

func (l *Ledger) writeLine(env envelope) error {
	f, err := os.OpenFile(l.path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return fmt.Errorf("ledger: open %s: %w", l.path, err)
	}
	defer f.Close()
	enc := json.NewEncoder(f)
	if err := enc.Encode(env); err != nil {
		return fmt.Errorf("ledger: append %s: %w", l.path, err)
	}
	return nil
}

func envelopeOf(kind string, v any) envelope {
	b, err := json.Marshal(v)
	if err != nil {
		panic(fmt.Sprintf("ledger: marshal %s: %v", kind, err))
	}
	return envelope{Kind: kind, Data: b}
}

// Decisions / Exposures / Feedback 快照视图（按 ID 排序，确定性）。
func (l *Ledger) Decisions() []contract.DecisionEvent {
	l.mu.Lock()
	defer l.mu.Unlock()
	out := make([]contract.DecisionEvent, 0, len(l.decisions))
	for _, id := range sortedKeys(l.decisions) {
		out = append(out, *l.decisions[id])
	}
	return out
}

func (l *Ledger) Exposures() []contract.ExposureEvent {
	l.mu.Lock()
	defer l.mu.Unlock()
	out := make([]contract.ExposureEvent, 0, len(l.exposures))
	for _, id := range sortedKeys(l.exposures) {
		out = append(out, *l.exposures[id])
	}
	return out
}

func (l *Ledger) Feedback() []contract.FeedbackEvent {
	l.mu.Lock()
	defer l.mu.Unlock()
	out := make([]contract.FeedbackEvent, 0, len(l.feedback))
	for _, id := range sortedKeys(l.feedback) {
		out = append(out, *l.feedback[id])
	}
	return out
}

func sortedKeys[T any](m map[string]*T) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

// DecisionFromDeal 把一个用户的发牌终稿 + 日志映射为 DecisionEvent
// （补位条目带 tag_overflow 原因码）。deck 未满、recent 过滤等排除性
// 原因由 deal.LogEntry 计数承载，进入 monitor 指标。
func DecisionFromDeal(
	user string, deck []recsys.DealRow, log deal.LogEntry,
	meta DecisionMeta,
) (contract.DecisionEvent, error) {
	if len(deck) == 0 {
		return contract.DecisionEvent{}, fmt.Errorf("ledger: empty deck for %s", user)
	}
	items := make([]contract.DecisionItem, 0, len(deck))
	for i, d := range deck {
		reason := contract.ReasonOK
		if log.Overflow && i >= log.AfterTagFilter {
			reason = contract.ReasonTagOverflow
		}
		items = append(items, contract.DecisionItem{ItemID: d.Item, Rank: d.Rank, Reason: reason})
	}
	ev := contract.DecisionEvent{
		DecisionID:        meta.DecisionID,
		RequestID:         meta.RequestID,
		SubjectID:         meta.SubjectID,
		OccurredAt:        meta.OccurredAt,
		ModelVersion:      meta.ModelVersion,
		FeatureSchemaHash: meta.FeatureSchemaHash,
		CandidateSetID:    meta.CandidateSetID,
		PolicyVersion:     meta.PolicyVersion,
		Items:             items,
	}
	if err := contract.ValidateDecision(&ev); err != nil {
		return contract.DecisionEvent{}, err
	}
	return ev, nil
}

// DecisionMeta 决策引用元数据。
type DecisionMeta struct {
	DecisionID        string
	RequestID         string
	SubjectID         string
	OccurredAt        time.Time
	ModelVersion      string
	FeatureSchemaHash string
	CandidateSetID    string
	PolicyVersion     string
}
