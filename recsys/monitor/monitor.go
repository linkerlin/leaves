// Package monitor 从账本与发牌工件聚合窗口指标并产出阈值状态
// （演进方案 §17.6 / RC-07）。只读账本；阈值由业务配置。
package monitor

import (
	"encoding/json"
	"fmt"
	"os"
	"time"

	"github.com/linkerlin/leaves/v2/recsys"
	"github.com/linkerlin/leaves/v2/recsys/contract"
	"github.com/linkerlin/leaves/v2/recsys/eval"
	"github.com/linkerlin/leaves/v2/recsys/ledger"
)

// Threshold 单指标阈值（与 eval.Threshold 同构，Level 决定越界级别）。
type Threshold = eval.Threshold

// Report monitor_report.json。
type Report struct {
	SchemaVersion int                `json:"schema_version"`
	WindowStart   time.Time          `json:"window_start"`
	WindowEnd     time.Time          `json:"window_end"`
	Metrics       map[string]float64 `json:"metrics"`
	States        []State            `json:"states"`
	Overall       string             `json:"overall"` // ok | warn | block | unevaluated
}

// State 单指标阈值结论（ok/warn/block + reason code）。
type State struct {
	Name   string  `json:"name"`
	Value  float64 `json:"value"`
	Status string  `json:"status"`
	Reason string  `json:"reason,omitempty"`
}

// Aggregate 聚合 [windowStart, windowEnd) 内账本指标与发牌质量。
// deals/catalogSize 传发牌终稿与目录规模（deck 层指标）。
func Aggregate(l *ledger.Ledger, windowStart, windowEnd time.Time, deals []recsys.DealRow, deckSize, maxSameTag, catalogSize int) map[string]float64 {
	m := map[string]float64{}
	exposures := l.Exposures()
	feedback := l.Feedback()

	shown, suppressed := 0, 0
	for _, x := range exposures {
		if x.OccurredAt.Before(windowStart) || !x.OccurredAt.Before(windowEnd) {
			continue
		}
		switch x.Status {
		case contract.ExposureShown:
			shown++
		case contract.ExposureSuppressed:
			suppressed++
		}
	}
	clicks := 0
	for _, f := range feedback {
		if f.OccurredAt.Before(windowStart) || !f.OccurredAt.Before(windowEnd) {
			continue
		}
		if f.EventType == contract.EventClick {
			clicks++
		}
	}
	m["exposure_count"] = float64(shown + suppressed)
	m["shown_count"] = float64(shown)
	m["suppressed_count"] = float64(suppressed)
	if shown > 0 {
		m["ctr"] = float64(clicks) / float64(shown)
	}
	// 完整性：账本已由 ledger 校验关联，孤立反馈（只挂 decision）比例。
	orphan := 0
	for _, f := range feedback {
		if f.ExposureID == "" {
			orphan++
		}
	}
	if len(feedback) > 0 {
		m["orphan_feedback_rate"] = float64(orphan) / float64(len(feedback))
	}

	if len(deals) > 0 && deckSize > 0 {
		fill, dup, overflow, itemCov := eval.DeckQuality(deals, deckSize, maxSameTag, catalogSize)
		m["deck_fill_rate"] = fill
		m["deck_dup_rate"] = dup
		m["deck_tag_overflow_rate"] = overflow
		m["deck_item_coverage"] = itemCov
	}
	return m
}

// Evaluate 评估阈值：block > warn > ok；未知指标 → block（integrity）。
// 无阈值 → overall=unevaluated（观察期仍需业务给出护栏）。
func Evaluate(metrics map[string]float64, thresholds []Threshold) []State {
	var states []State
	for _, th := range thresholds {
		v, ok := metrics[th.Name]
		st := State{Name: th.Name, Value: v, Status: contract.StatusOK}
		switch {
		case !ok:
			st.Status = contract.StatusBlock
			st.Reason = "metric_not_computed"
		case th.Min != nil && v < *th.Min, th.Max != nil && v > *th.Max:
			st.Status = th.Level
			st.Reason = fmt.Sprintf("value %v outside configured bound", v)
		}
		states = append(states, st)
	}
	return states
}

// Overall 取最差状态。
func Overall(states []State) string {
	worst := "unevaluated"
	rank := map[string]int{contract.StatusOK: 0, contract.StatusWarn: 1, contract.StatusBlock: 2, "unevaluated": -1}
	for _, s := range states {
		if rank[s.Status] > rank[worst] {
			worst = s.Status
		}
	}
	return worst
}

// BuildReport 聚合 + 评估一步完成。
func BuildReport(l *ledger.Ledger, windowStart, windowEnd time.Time, deals []recsys.DealRow, deckSize, maxSameTag, catalogSize int, thresholds []Threshold) *Report {
	r := &Report{
		SchemaVersion: contract.SchemaVersion,
		WindowStart:   windowStart,
		WindowEnd:     windowEnd,
		Metrics:       Aggregate(l, windowStart, windowEnd, deals, deckSize, maxSameTag, catalogSize),
	}
	r.States = Evaluate(r.Metrics, thresholds)
	r.Overall = Overall(r.States)
	return r
}

// WriteReport 写 monitor_report.json。
func WriteReport(path string, r *Report) error {
	b, err := json.MarshalIndent(r, "", "  ")
	if err != nil {
		return fmt.Errorf("monitor: marshal: %w", err)
	}
	if err := os.WriteFile(path, b, 0o644); err != nil {
		return fmt.Errorf("monitor: write %s: %w", path, err)
	}
	return nil
}
