package monitor

import (
	"fmt"
	"time"

	"github.com/linkerlin/leaves/v2/recsys/contract"
)

// 触发动作（对应 release 状态机的请求入口）。
const (
	ActionRetrain  = "retrain_requested"
	ActionRollback = "rollback_requested"
)

// severity 越界级别排序：ok < warn < block。
var severity = map[string]int{
	contract.StatusOK:    0,
	contract.StatusWarn:  1,
	contract.StatusBlock: 2,
}

// Trigger 可配置触发规则（演进方案 §17.6：触发器是带窗口与冷却期的规则，
// 不是 Agent 的临场猜测）：指标连续 N 个窗口处于越界级别 → 动作。
// 安全类规则用 Level=block + Consecutive=1（立即触发）。
type Trigger struct {
	Name        string        `json:"name"`
	Metric      string        `json:"metric"`              // Report.States 中的指标名
	Level       string        `json:"level"`               // 越界级别：warn | block
	Consecutive int           `json:"consecutive_windows"` // 连续越界窗口数（<=1 视为立即）
	Action      string        `json:"action"`              // retrain_requested | rollback_requested
	Cooldown    time.Duration `json:"cooldown"`            // 同一规则两次触发的最小间隔（0 = 不限）
}

// Fired 一次已触发的规则。
type Fired struct {
	Rule   string `json:"rule"`
	Metric string `json:"metric"`
	Action string `json:"action"`
	Reason string `json:"reason"`
	Streak int    `json:"streak"` // 触发时已连续越界的窗口数
}

type triggerState struct {
	streak    int
	lastFired time.Time
}

// TriggerSet 有状态触发器集合：跨窗口累计越界序列、记录冷却。
// 单 goroutine 使用；状态不持久化（重启后从当前窗口重新累计）。
type TriggerSet struct {
	rules  []Trigger
	states map[string]*triggerState
}

// NewTriggerSet 校验并构建触发器集合。
func NewTriggerSet(rules []Trigger) (*TriggerSet, error) {
	seen := map[string]bool{}
	ts := &TriggerSet{rules: append([]Trigger(nil), rules...), states: map[string]*triggerState{}}
	for i, r := range ts.rules {
		if r.Name == "" || r.Metric == "" {
			return nil, fmt.Errorf("monitor: trigger[%d] name/metric is empty", i)
		}
		if seen[r.Name] {
			return nil, fmt.Errorf("monitor: duplicate trigger name %s", r.Name)
		}
		seen[r.Name] = true
		switch r.Level {
		case contract.StatusWarn, contract.StatusBlock:
		default:
			return nil, fmt.Errorf("monitor: trigger %s level must be warn|block, got %q", r.Name, r.Level)
		}
		switch r.Action {
		case ActionRetrain, ActionRollback:
		default:
			return nil, fmt.Errorf("monitor: trigger %s unknown action %q", r.Name, r.Action)
		}
		if r.Consecutive < 1 {
			ts.rules[i].Consecutive = 1
		}
		if r.Cooldown < 0 {
			return nil, fmt.Errorf("monitor: trigger %s negative cooldown", r.Name)
		}
		ts.states[r.Name] = &triggerState{}
	}
	return ts, nil
}

// Evaluate 喂入一个窗口报告，返回本窗口触发的规则（可为空）。
// 未配置阈值的指标（States 中缺失）不计越界并重置该规则的连续计数。
func (ts *TriggerSet) Evaluate(rep *Report, now time.Time) []Fired {
	stateByMetric := map[string]State{}
	for _, s := range rep.States {
		stateByMetric[s.Name] = s
	}
	var fired []Fired
	for i := range ts.rules {
		r := &ts.rules[i]
		st := ts.states[r.Name]
		s, ok := stateByMetric[r.Metric]
		if !ok || severity[s.Status] < severity[r.Level] {
			st.streak = 0
			continue
		}
		st.streak++
		if st.streak < r.Consecutive {
			continue
		}
		if r.Cooldown > 0 && !st.lastFired.IsZero() && now.Sub(st.lastFired) < r.Cooldown {
			continue // 冷却期内抑制；streak 保留，冷却结束且仍越界时会再触发
		}
		st.lastFired = now
		fired = append(fired, Fired{
			Rule: r.Name, Metric: r.Metric, Action: r.Action,
			Reason: fmt.Sprintf("%s %s for %d consecutive window(s): %s", r.Metric, s.Status, st.streak, s.Reason),
			Streak: st.streak,
		})
		st.streak = 0
	}
	return fired
}
