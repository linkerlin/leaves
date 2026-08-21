// Package split 提供带 as-of 语义的时间切分（演进方案 §17.5 / RC-03）。
//
// 默认主线为时间切分：train_end < validation_start < test_start；
// 训练样本不得读取 validation/test 起点之后的事件。用户切分仅保留为
// cold-start 附加实验（ColdStartUsers），不能替代时间评估。
package split

import (
	"fmt"
	"sort"
	"time"

	"github.com/linkerlin/leaves/v2/recsys"
	"github.com/linkerlin/leaves/v2/recsys/contract"
)

// TimeConfig 时间切分边界（UTC、半开区间）。
// train = [-, TrainEnd)；validation = [ValStart, TestStart)；test = [TestStart, -)。
// [TrainEnd, ValStart) 为隔离带，事件丢弃以防泄漏。
type TimeConfig struct {
	TrainEnd  time.Time
	ValStart  time.Time
	TestStart time.Time
}

// Validate 边界顺序与非零校验。
func (c TimeConfig) Validate() error {
	for name, t := range map[string]time.Time{"train_end": c.TrainEnd, "validation_start": c.ValStart, "test_start": c.TestStart} {
		if t.IsZero() {
			return fmt.Errorf("split: %s is zero", name)
		}
		if t.Location() != time.UTC {
			return fmt.Errorf("split: %s must be UTC", name)
		}
	}
	if !c.TrainEnd.Before(c.ValStart) || !c.ValStart.Before(c.TestStart) {
		return fmt.Errorf("split: require train_end < validation_start < test_start (got %s / %s / %s)",
			c.TrainEnd, c.ValStart, c.TestStart)
	}
	return nil
}

// Split 按时间边界切分交互事件。先 ValidateInteractions 再切分。
func Split(events []contract.InteractionEvent, cfg TimeConfig) (train, val, test []contract.InteractionEvent, err error) {
	if err := cfg.Validate(); err != nil {
		return nil, nil, nil, err
	}
	if err := contract.ValidateInteractions(events); err != nil {
		return nil, nil, nil, err
	}
	for _, e := range events {
		switch {
		case e.OccurredAt.Before(cfg.TrainEnd):
			train = append(train, e)
		case !e.OccurredAt.Before(cfg.ValStart) && e.OccurredAt.Before(cfg.TestStart):
			val = append(val, e)
		case !e.OccurredAt.Before(cfg.TestStart):
			test = append(test, e)
		}
	}
	return train, val, test, nil
}

// SuggestTimeConfig 按事件时间分位确定性推导边界：train=[…,p70)、
// 隔离带 [p70, p70+1ms)、val=[p70+1ms, p85+1ms)、test=[p85+1ms,…)。
// 事件数 <10 时拒绝（分位无意义）。
func SuggestTimeConfig(events []contract.InteractionEvent) (TimeConfig, error) {
	if len(events) < 10 {
		return TimeConfig{}, fmt.Errorf("split: need >=10 events to suggest boundaries (got %d)", len(events))
	}
	ts := make([]time.Time, len(events))
	for i := range events {
		ts[i] = events[i].OccurredAt
	}
	sort.Slice(ts, func(i, j int) bool { return ts[i].Before(ts[j]) })
	at := func(p float64) time.Time { return ts[int(p*float64(len(ts)-1))] }
	trainEnd := at(0.70)
	return TimeConfig{
		TrainEnd:  trainEnd,
		ValStart:  trainEnd.Add(time.Millisecond),
		TestStart: at(0.85).Add(time.Millisecond),
	}, nil
}

// CheckLeakage 断言训练事件全部早于 evalStart（as-of 因果门禁）。
func CheckLeakage(train []contract.InteractionEvent, evalStart time.Time) error {
	for _, e := range train {
		if !e.OccurredAt.Before(evalStart) {
			return fmt.Errorf("split: leakage: train event %s at %s >= eval start %s", e.EventID, e.OccurredAt, evalStart)
		}
	}
	return nil
}

// ColdStartUsers 返回 eval 切中未在 train 出现过的 subject（cold-start 分层）。
// 附加实验用途；不得替代时间切分主线。
func ColdStartUsers(train, eval []contract.InteractionEvent) map[string]bool {
	seen := make(map[string]bool, len(train))
	for _, e := range train {
		seen[e.SubjectID] = true
	}
	cold := map[string]bool{}
	for _, e := range eval {
		if !seen[e.SubjectID] {
			cold[e.SubjectID] = true
		}
	}
	return cold
}

// TimeRange 计算事件批的 [min, max) 时间范围（空批返回零值）。
func TimeRange(events []contract.InteractionEvent) contract.TimeRange {
	if len(events) == 0 {
		return contract.TimeRange{}
	}
	ts := make([]time.Time, len(events))
	for i, e := range events {
		ts[i] = e.OccurredAt
	}
	sort.Slice(ts, func(i, j int) bool { return ts[i].Before(ts[j]) })
	return contract.TimeRange{Start: ts[0], End: ts[len(ts)-1].Add(time.Millisecond)}
}

// Assign 把交互事件映射为旧四元 Interaction 供离线四段流水线复用；
// 事件顺序按 (occurred_at, event_id) 稳定排序（确定性回放的前提）。
func Assign(events []contract.InteractionEvent) []recsys.Interaction {
	sorted := make([]contract.InteractionEvent, len(events))
	copy(sorted, events)
	sort.Slice(sorted, func(i, j int) bool {
		if sorted[i].OccurredAt.Equal(sorted[j].OccurredAt) {
			return sorted[i].EventID < sorted[j].EventID
		}
		return sorted[i].OccurredAt.Before(sorted[j].OccurredAt)
	})
	out := make([]recsys.Interaction, len(sorted))
	for i, e := range sorted {
		out[i] = recsys.Interaction{User: e.SubjectID, Item: e.ItemID, Score: e.Value}
	}
	return out
}
