// Package replay 从决策/曝光/反馈账本重建训练样本（演进方案 §17.6 / RC-06）。
//
// 规则：只有可归因到 shown 曝光的反馈才构成 supervised 正样本；
// 曝光后无反馈（或反馈超出归因窗）按负样本策略决定是否成负样本；
// 迟到/孤立/抑制曝光上的反馈计数入报告，不进样本。确定性：同输入同输出。
package replay

import (
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"time"

	"github.com/linkerlin/leaves/v2/recsys/contract"
	"github.com/linkerlin/leaves/v2/recsys/ledger"
)

// NegativePolicy 负样本策略。
const (
	NegativeNone            = "none"              // 只产生正样本
	NegativeImpressedNoFeed = "impressed_no_feed" // shown 曝光无窗内反馈 → 负样本
)

// Config 回放配置。
type Config struct {
	AttributionWindow time.Duration `json:"attribution_window"`
	NegativePolicy    string        `json:"negative_policy"`
	// PositiveThreshold：反馈值 >= 该阈值记正样本（label=反馈值）；默认 >0 即正。
	PositiveThreshold *float64 `json:"positive_threshold,omitempty"`
}

// DefaultConfig 默认 24h 归因窗 + impressed_no_feed 负样本。
func DefaultConfig() Config {
	return Config{AttributionWindow: 24 * time.Hour, NegativePolicy: NegativeImpressedNoFeed}
}

// Sample 回放出的训练样本（可直接进 split/Assign 流程）。
type Sample struct {
	SubjectID  string    `json:"subject_id"`
	ItemID     string    `json:"item_id"`
	Label      float64   `json:"label"`
	ExposureID string    `json:"exposure_id"`
	DecisionID string    `json:"decision_id"`
	ExposureAt time.Time `json:"exposure_at"`
}

// Report replay_report.json：样本构成与差异原因。
type Report struct {
	SchemaVersion      int      `json:"schema_version"`
	Positives          int      `json:"positives"`
	Negatives          int      `json:"negatives"`
	LateFeedback       int      `json:"late_feedback"`
	OrphanFeedback     int      `json:"orphan_feedback"`
	SuppressedFeedback int      `json:"suppressed_feedback"`
	DedupedFeedback    int      `json:"deduped_feedback"`
	DiffReasons        []string `json:"diff_reasons"`
}

// BuildSamples 归因窗内样本重建。
func BuildSamples(l *ledger.Ledger, cfg Config) ([]Sample, Report, error) {
	if cfg.AttributionWindow <= 0 {
		return nil, Report{}, fmt.Errorf("replay: attribution window must be positive")
	}
	switch cfg.NegativePolicy {
	case NegativeNone, NegativeImpressedNoFeed:
	default:
		return nil, Report{}, fmt.Errorf("replay: unknown negative policy %q", cfg.NegativePolicy)
	}
	rep := Report{SchemaVersion: contract.SchemaVersion}

	exposures := l.Exposures()
	feedback := l.Feedback()
	decisions := l.Decisions()
	decisionOf := make(map[string]contract.DecisionEvent, len(decisions))
	for _, d := range decisions {
		decisionOf[d.DecisionID] = d
	}
	itemOf := map[string]string{}
	for _, d := range decisions {
		for _, it := range d.Items {
			itemOf[d.DecisionID+"\x00"+it.ItemID] = it.ItemID
		}
	}

	// 反馈按曝光聚合（确定性排序）。
	feedByExposure := map[string][]contract.FeedbackEvent{}
	for _, f := range feedback {
		if f.ExposureID == "" {
			rep.OrphanFeedback++
			continue
		}
		feedByExposure[f.ExposureID] = append(feedByExposure[f.ExposureID], f)
	}

	var samples []Sample
	for _, x := range exposures {
		if x.Status != contract.ExposureShown {
			if len(feedByExposure[x.ExposureID]) > 0 {
				rep.SuppressedFeedback++
			}
			continue
		}
		d, ok := decisionOf[x.DecisionID]
		if !ok || itemOf[x.DecisionID+"\x00"+x.ItemID] == "" {
			return nil, Report{}, fmt.Errorf("replay: exposure %s lost decision linkage", x.ExposureID)
		}
		subject := subjectOf(d, x.ItemID)

		// 窗内反馈取值最大的一条为正样本；窗外为迟到。
		var label float64
		var positive bool
		windowEnd := x.OccurredAt.Add(cfg.AttributionWindow)
		inWindow := 0
		for _, f := range feedByExposure[x.ExposureID] {
			if f.OccurredAt.After(windowEnd) {
				rep.LateFeedback++
				continue
			}
			inWindow++
			if f.Value > label {
				label = f.Value
			}
		}
		if inWindow > 1 {
			rep.DedupedFeedback += inWindow - 1
		}
		if thr := cfg.PositiveThreshold; thr != nil {
			positive = label >= *thr
		} else {
			positive = label > 0
		}

		switch {
		case positive:
			rep.Positives++
			samples = append(samples, Sample{
				SubjectID: subject, ItemID: x.ItemID, Label: label,
				ExposureID: x.ExposureID, DecisionID: x.DecisionID, ExposureAt: x.OccurredAt,
			})
		case cfg.NegativePolicy == NegativeImpressedNoFeed:
			rep.Negatives++
			samples = append(samples, Sample{
				SubjectID: subject, ItemID: x.ItemID, Label: 0,
				ExposureID: x.ExposureID, DecisionID: x.DecisionID, ExposureAt: x.OccurredAt,
			})
		}
	}

	sort.Slice(samples, func(i, j int) bool {
		if samples[i].ExposureAt.Equal(samples[j].ExposureAt) {
			return samples[i].ExposureID < samples[j].ExposureID
		}
		return samples[i].ExposureAt.Before(samples[j].ExposureAt)
	})

	rep.DiffReasons = diffReasons(rep)
	return samples, rep, nil
}

// subjectOf 反查匿名 subject；决策未携带 subject_id 时退回 decision_id
// （保持可追溯，不猜匿名身份）。
func subjectOf(d contract.DecisionEvent, item string) string {
	if d.SubjectID != "" {
		return d.SubjectID
	}
	return d.DecisionID
}

func diffReasons(r Report) []string {
	var out []string
	if r.OrphanFeedback > 0 {
		out = append(out, fmt.Sprintf("orphan_feedback_excluded:%d", r.OrphanFeedback))
	}
	if r.LateFeedback > 0 {
		out = append(out, fmt.Sprintf("late_feedback_excluded:%d", r.LateFeedback))
	}
	if r.SuppressedFeedback > 0 {
		out = append(out, fmt.Sprintf("suppressed_feedback_excluded:%d", r.SuppressedFeedback))
	}
	if r.DedupedFeedback > 0 {
		out = append(out, fmt.Sprintf("duplicate_feedback_deduped:%d", r.DedupedFeedback))
	}
	return out
}

// WriteReport 写 replay_report.json。
func WriteReport(path string, r Report) error {
	b, err := json.MarshalIndent(r, "", "  ")
	if err != nil {
		return fmt.Errorf("replay: marshal: %w", err)
	}
	if err := os.WriteFile(path, b, 0o644); err != nil {
		return fmt.Errorf("replay: write %s: %w", path, err)
	}
	return nil
}
