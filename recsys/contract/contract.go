// Package contract 冻结推荐生产闭环的核心契约（演进方案 §17.4）：
// 数据快照、交互事件、决策/曝光/反馈事件与发布证据。
//
// 不变量：
//   - 所有事件携带 UTC 时间与 schema 版本；
//   - 所有 ID 为可匿名化的稳定字符串，不含原始 PII；
//   - 字段只增不删；旧四元数据经 LegacySnapshot 显式导入，不进入时间因果门禁。
package contract

import (
	"encoding/hex"
	"fmt"
	"strings"
	"time"
)

// SchemaVersion 当前契约 schema 版本。
const SchemaVersion = 1

// EventType 交互/反馈事件类型枚举。
type EventType string

const (
	EventImpression EventType = "impression"
	EventClick      EventType = "click"
	EventConversion EventType = "conversion"
	EventRating     EventType = "rating"
)

// FeedbackTypes 可作为 FeedbackEvent 类型的事件子集（impression 属于曝光层）。
var FeedbackTypes = map[EventType]bool{
	EventClick:      true,
	EventConversion: true,
	EventRating:     true,
}

// 决策条目原因码：Tag 超限补位、候选不足、fallback、过滤命中必须显式携带。
const (
	ReasonOK                     = "ok"
	ReasonTagOverflow            = "tag_overflow"
	ReasonInsufficientCandidates = "insufficient_candidates"
	ReasonFallback               = "fallback"
	ReasonFiltered               = "filtered"
)

// KnownReasons 已知原因码集合。
var KnownReasons = map[string]bool{
	ReasonOK:                     true,
	ReasonTagOverflow:            true,
	ReasonInsufficientCandidates: true,
	ReasonFallback:               true,
	ReasonFiltered:               true,
}

// 门禁状态。
const (
	StatusOK    = "ok"
	StatusWarn  = "warn"
	StatusBlock = "block"
)

// 曝光展示状态。
const (
	ExposureShown      = "shown"
	ExposureSuppressed = "suppressed"
)

// FileRef 输入文件引用：路径 + 内容 sha256。
type FileRef struct {
	Path   string `json:"path"`
	SHA256 string `json:"sha256"`
}

// TimeRange 数据时间范围（UTC 闭开区间语义：[Start, End)）。
type TimeRange struct {
	Start time.Time `json:"start"`
	End   time.Time `json:"end"`
}

// DatasetSnapshot 不可变数据快照。训练、评估、发布必须引用同一快照。
type DatasetSnapshot struct {
	SnapshotID        string    `json:"snapshot_id"`
	SchemaVersion     int       `json:"schema_version"`
	CreatedAt         time.Time `json:"created_at"`
	Purpose           string    `json:"purpose"` // train | eval | release
	InputFiles        []FileRef `json:"input_files"`
	FeatureSchemaHash string    `json:"feature_schema_hash"`
	TimeRange         TimeRange `json:"time_range"`
	LegacySnapshot    bool      `json:"legacy_snapshot,omitempty"`
}

// InteractionEvent 带时间语义的交互事件（impression/click/conversion/rating）。
type InteractionEvent struct {
	EventID    string    `json:"event_id"`
	OccurredAt time.Time `json:"occurred_at"`
	SubjectID  string    `json:"subject_id"`
	ItemID     string    `json:"item_id"`
	EventType  EventType `json:"event_type"`
	Value      float64   `json:"value"`
	Source     string    `json:"source"`
}

// DecisionItem 决策 Top-K 条目；Reason 为原因码（审计真源）。
type DecisionItem struct {
	ItemID string `json:"item_id"`
	Rank   int    `json:"rank"`
	Reason string `json:"reason"`
}

// DecisionEvent 一次排序+发牌决策。DealRow 保留为展示行，本结构才是审计真源。
type DecisionEvent struct {
	DecisionID        string         `json:"decision_id"`
	RequestID         string         `json:"request_id"`
	SubjectID         string         `json:"subject_id,omitempty"` // 匿名键；反馈回放的 join 依据
	OccurredAt        time.Time      `json:"occurred_at"`
	ModelVersion      string         `json:"model_version"`
	FeatureSchemaHash string         `json:"feature_schema_hash"`
	CandidateSetID    string         `json:"candidate_set_id"`
	PolicyVersion     string         `json:"policy_version"`
	Items             []DecisionItem `json:"items"`
}

// ExposureEvent 一次曝光；反馈归因必须先有曝光。
type ExposureEvent struct {
	ExposureID string    `json:"exposure_id"`
	DecisionID string    `json:"decision_id"`
	ItemID     string    `json:"item_id"`
	Position   int       `json:"position"`
	OccurredAt time.Time `json:"occurred_at"`
	Status     string    `json:"status"` // shown | suppressed
}

// FeedbackEvent 用户反馈；未关联曝光的反馈默认不作为 supervised 正样本。
type FeedbackEvent struct {
	EventID     string    `json:"event_id"`
	ExposureID  string    `json:"exposure_id,omitempty"`
	DecisionID  string    `json:"decision_id,omitempty"`
	OccurredAt  time.Time `json:"occurred_at"`
	EventType   EventType `json:"event_type"`
	Value       float64   `json:"value"`
	Attribution string    `json:"attribution_window_version,omitempty"`
}

// GateResult 单层门禁结论（数据/候选排序/发牌）。
type GateResult struct {
	Layer  string `json:"layer"` // data | candidate_rank | deal
	Name   string `json:"name"`
	Status string `json:"status"` // ok | warn | block
	Reason string `json:"reason,omitempty"`
}

// ReleaseEvidence 发布证据包：候选模型、快照、指标、门禁、审批与回滚目标。
type ReleaseEvidence struct {
	ReleaseID      string             `json:"release_id"`
	ModelSHA256    string             `json:"model_sha256"`
	QuantSHA256    string             `json:"quant_sha256,omitempty"`
	RunID          string             `json:"run_id"`
	SnapshotID     string             `json:"snapshot_id"`
	PolicyVersion  string             `json:"policy_version"`
	OfflineMetrics map[string]float64 `json:"offline_metrics"`
	OnlineMetrics  map[string]float64 `json:"online_metrics,omitempty"`
	Gates          []GateResult       `json:"gate_results"`
	ApprovedBy     string             `json:"approved_by,omitempty"`
	RollbackTarget string             `json:"rollback_target,omitempty"`
	CreatedAt      time.Time          `json:"created_at"`
}

func validHash(h string) bool {
	if len(h) != 64 {
		return false
	}
	_, err := hex.DecodeString(h)
	return err == nil
}

func checkUTC(field string, t time.Time) error {
	if t.IsZero() {
		return fmt.Errorf("contract: %s time is zero", field)
	}
	if t.Location() != time.UTC {
		return fmt.Errorf("contract: %s time must be UTC (got %s)", field, t.Location())
	}
	return nil
}

// checkAnonID 匿名键防线：拒绝空白与 '@'（原始邮箱混入的绊线）。
func checkAnonID(field, v string) error {
	if v == "" {
		return fmt.Errorf("contract: %s is empty", field)
	}
	if strings.ContainsAny(v, " \t\r\n@") {
		return fmt.Errorf("contract: %s must be an anonymized stable key (no whitespace/'@'): %q", field, v)
	}
	return nil
}

// ValidateSnapshot 校验数据快照完整性与 hash 格式。
func ValidateSnapshot(s *DatasetSnapshot) error {
	if s.SnapshotID == "" {
		return fmt.Errorf("contract: snapshot_id is empty")
	}
	if s.SchemaVersion != SchemaVersion {
		return fmt.Errorf("contract: schema_version %d != %d", s.SchemaVersion, SchemaVersion)
	}
	if err := checkUTC("created_at", s.CreatedAt); err != nil {
		return err
	}
	if s.Purpose == "" {
		return fmt.Errorf("contract: purpose is empty")
	}
	for i, f := range s.InputFiles {
		if f.Path == "" {
			return fmt.Errorf("contract: input_files[%d].path is empty", i)
		}
		if !validHash(f.SHA256) {
			return fmt.Errorf("contract: input_files[%d].sha256 invalid: %q", i, f.SHA256)
		}
	}
	if !s.LegacySnapshot && !validHash(s.FeatureSchemaHash) {
		return fmt.Errorf("contract: feature_schema_hash invalid (legacy 数据须显式 LegacySnapshot)")
	}
	if err := checkUTC("time_range.start", s.TimeRange.Start); err != nil {
		return err
	}
	if !s.TimeRange.End.IsZero() && s.TimeRange.End.Before(s.TimeRange.Start) {
		return fmt.Errorf("contract: time_range end before start")
	}
	return nil
}

// ValidateInteractions 校验交互事件批：ID 唯一、类型已知、UTC、匿名键。
func ValidateInteractions(events []InteractionEvent) error {
	seen := make(map[string]bool, len(events))
	for i := range events {
		e := &events[i]
		if e.EventID == "" {
			return fmt.Errorf("contract: events[%d].event_id is empty", i)
		}
		if seen[e.EventID] {
			return fmt.Errorf("contract: duplicate event_id %s", e.EventID)
		}
		seen[e.EventID] = true
		if err := checkUTC("occurred_at", e.OccurredAt); err != nil {
			return fmt.Errorf("%s (event %s)", err, e.EventID)
		}
		if err := checkAnonID("subject_id", e.SubjectID); err != nil {
			return fmt.Errorf("%s (event %s)", err, e.EventID)
		}
		if err := checkAnonID("item_id", e.ItemID); err != nil {
			return fmt.Errorf("%s (event %s)", err, e.EventID)
		}
		if e.EventType == "" {
			return fmt.Errorf("contract: events[%d].event_type is empty", i)
		}
		switch e.EventType {
		case EventImpression, EventClick, EventConversion, EventRating:
		default:
			return fmt.Errorf("contract: unknown event_type %q (event %s)", e.EventType, e.EventID)
		}
	}
	return nil
}

// ValidateDecision 校验决策事件：引用齐全、Top-K 条目带原因码。
func ValidateDecision(d *DecisionEvent) error {
	if d.DecisionID == "" || d.RequestID == "" {
		return fmt.Errorf("contract: decision_id/request_id is empty")
	}
	if err := checkUTC("occurred_at", d.OccurredAt); err != nil {
		return err
	}
	if d.ModelVersion == "" || d.CandidateSetID == "" || d.PolicyVersion == "" {
		return fmt.Errorf("contract: model/candidate_set/policy version is empty (decision %s)", d.DecisionID)
	}
	if d.SubjectID != "" {
		if err := checkAnonID("subject_id", d.SubjectID); err != nil {
			return fmt.Errorf("%s (decision %s)", err, d.DecisionID)
		}
	}
	if !validHash(d.FeatureSchemaHash) {
		return fmt.Errorf("contract: feature_schema_hash invalid (decision %s)", d.DecisionID)
	}
	if len(d.Items) == 0 {
		return fmt.Errorf("contract: decision %s has no items", d.DecisionID)
	}
	seenItem := map[string]bool{}
	seenRank := map[int]bool{}
	for i, it := range d.Items {
		if err := checkAnonID(fmt.Sprintf("items[%d].item_id", i), it.ItemID); err != nil {
			return fmt.Errorf("%s (decision %s)", err, d.DecisionID)
		}
		if seenItem[it.ItemID] {
			return fmt.Errorf("contract: duplicate item %s (decision %s)", it.ItemID, d.DecisionID)
		}
		seenItem[it.ItemID] = true
		if it.Rank < 1 || seenRank[it.Rank] {
			return fmt.Errorf("contract: items[%d].rank %d invalid or duplicated (decision %s)", i, it.Rank, d.DecisionID)
		}
		seenRank[it.Rank] = true
		if !KnownReasons[it.Reason] {
			return fmt.Errorf("contract: items[%d].reason %q unknown (decision %s)", i, it.Reason, d.DecisionID)
		}
	}
	return nil
}

// ValidateExposure 校验曝光事件并回链决策：时间不得早于决策、条目须在 Top-K。
func ValidateExposure(e *ExposureEvent, d *DecisionEvent) error {
	if e.ExposureID == "" || e.DecisionID == "" || e.ItemID == "" {
		return fmt.Errorf("contract: exposure_id/decision_id/item_id is empty")
	}
	if e.Position < 1 {
		return fmt.Errorf("contract: exposure %s position %d < 1", e.ExposureID, e.Position)
	}
	if e.Status != ExposureShown && e.Status != ExposureSuppressed {
		return fmt.Errorf("contract: exposure %s unknown status %q", e.ExposureID, e.Status)
	}
	if err := checkUTC("occurred_at", e.OccurredAt); err != nil {
		return err
	}
	if d == nil {
		return fmt.Errorf("contract: exposure %s references unknown decision %s", e.ExposureID, e.DecisionID)
	}
	if d.DecisionID != e.DecisionID {
		return fmt.Errorf("contract: exposure %s decision mismatch (%s != %s)", e.ExposureID, e.DecisionID, d.DecisionID)
	}
	if e.OccurredAt.Before(d.OccurredAt) {
		return fmt.Errorf("contract: exposure %s occurred before decision (reverse time)", e.ExposureID)
	}
	for _, it := range d.Items {
		if it.ItemID == e.ItemID {
			if it.Rank != e.Position {
				return fmt.Errorf("contract: exposure %s position %d != decision rank %d", e.ExposureID, e.Position, it.Rank)
			}
			return nil
		}
	}
	return fmt.Errorf("contract: exposure %s item %s not in decision %s", e.ExposureID, e.ItemID, e.DecisionID)
}

// ValidateFeedback 校验反馈事件并回链曝光：时间不得早于曝光、类型须为反馈子集。
// exposure 可为 nil 仅当反馈只挂 DecisionID（该类反馈不构成 supervised 正样本）。
func ValidateFeedback(f *FeedbackEvent, exposure *ExposureEvent) error {
	if f.EventID == "" {
		return fmt.Errorf("contract: feedback event_id is empty")
	}
	if err := checkUTC("occurred_at", f.OccurredAt); err != nil {
		return err
	}
	if !FeedbackTypes[f.EventType] {
		return fmt.Errorf("contract: feedback %s type %q not in feedback subset", f.EventID, f.EventType)
	}
	if f.ExposureID == "" && f.DecisionID == "" {
		return fmt.Errorf("contract: feedback %s has no exposure/decision reference", f.EventID)
	}
	if f.ExposureID != "" {
		if exposure == nil {
			return fmt.Errorf("contract: feedback %s references unknown exposure %s", f.EventID, f.ExposureID)
		}
		if exposure.ExposureID != f.ExposureID {
			return fmt.Errorf("contract: feedback %s exposure mismatch (%s != %s)", f.EventID, f.ExposureID, exposure.ExposureID)
		}
		if f.OccurredAt.Before(exposure.OccurredAt) {
			return fmt.Errorf("contract: feedback %s occurred before exposure (reverse time)", f.EventID)
		}
	}
	return nil
}

// ValidateEvidence 校验发布证据：模型 hash、门禁结论、时间齐全。
func ValidateEvidence(ev *ReleaseEvidence) error {
	if ev.ReleaseID == "" || ev.RunID == "" || ev.SnapshotID == "" || ev.PolicyVersion == "" {
		return fmt.Errorf("contract: release_id/run_id/snapshot_id/policy_version is empty")
	}
	if !validHash(ev.ModelSHA256) {
		return fmt.Errorf("contract: model_sha256 invalid")
	}
	if ev.QuantSHA256 != "" && !validHash(ev.QuantSHA256) {
		return fmt.Errorf("contract: quant_sha256 invalid")
	}
	if err := checkUTC("created_at", ev.CreatedAt); err != nil {
		return err
	}
	for i, g := range ev.Gates {
		switch g.Status {
		case StatusOK, StatusWarn, StatusBlock:
		default:
			return fmt.Errorf("contract: gate_results[%d].status %q unknown", i, g.Status)
		}
		if g.Layer == "" || g.Name == "" {
			return fmt.Errorf("contract: gate_results[%d].layer/name is empty", i)
		}
	}
	return nil
}
