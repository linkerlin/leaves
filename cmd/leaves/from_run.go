package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

// loadRunFromLedger 从 runs.jsonl 选取一行：
//   - tag 非空：取该 tag 的最后一次出现（同 tag 重跑时以最新为准）
//   - tag 为空：按 maximize 取 value 最优行
func loadRunFromLedger(path, tag string) (*runRecord, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, errAgentWrap("data_load", fmt.Sprintf("open --from-run: %v", err),
			"确认 runs.jsonl 路径存在且由 train --runs 写出", false, err)
	}
	defer f.Close()

	var (
		byTag    *runRecord
		best     *runRecord
		bestInit bool
		n        int
	)
	sc := bufio.NewScanner(f)
	// 单行可能较长（完整 params）；抬高 buffer。
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		var rec runRecord
		if err := json.Unmarshal([]byte(line), &rec); err != nil {
			return nil, errAgent("data_load", fmt.Sprintf("runs.jsonl 第 %d 行 JSON 无效: %v", n+1, err),
				"账本须由 leaves train --runs 追加，勿手写损坏行", false)
		}
		n++
		if tag != "" {
			if rec.Tag == tag {
				cp := rec
				byTag = &cp
			}
			continue
		}
		// 无 tag：按 maximize 选最优
		cp := rec
		if !bestInit {
			best = &cp
			bestInit = true
			continue
		}
		if betterRun(&cp, best) {
			best = &cp
		}
	}
	if err := sc.Err(); err != nil {
		return nil, fmt.Errorf("read runs: %w", err)
	}
	if n == 0 {
		return nil, errAgent("data_load", "runs.jsonl 为空",
			"先 train --runs PATH --tag NAME 写入至少一行", false)
	}
	if tag != "" {
		if byTag == nil {
			return nil, errAgent("usage", fmt.Sprintf("runs.jsonl 中无 tag=%q", tag),
				"用 Get-Content runs.jsonl 查看已有 tag；或不带 --tag 自动选最优", false)
		}
		return byTag, nil
	}
	return best, nil
}

func betterRun(a, b *runRecord) bool {
	if a == nil {
		return false
	}
	if b == nil {
		return true
	}
	if a.Maximize {
		return a.Value > b.Value
	}
	return a.Value < b.Value
}

// applyParamsIfUnset 把账本 params 填入未在 CLI 显式设置的 flag（CLI 覆盖优先）。
// set 由 flag.FlagSet.Visit 收集的显式 flag 名。
func applyParamsIfUnset(
	set map[string]bool,
	p *paramsRecord,
	objective *string,
	evalMetric *string,
	numClass, rounds, depth, maxLeaves *int,
	lr, lambda, minChildWeight, gamma *float64,
	maxBin *int,
	subsample, colsample *float64,
	treeMethod *string,
	ndcgK *int,
	cv, earlyStop *int,
	seed *int64,
	runObjective string,
) {
	if p == nil {
		// 仍可从 run 行顶层 objective 补
		if !set["objective"] && runObjective != "" && *objective == "" {
			*objective = runObjective
		}
		return
	}
	// objective：账本行顶层优先（WP-17 写入）；params 不含 objective。
	if !set["objective"] && *objective == "" && runObjective != "" {
		*objective = runObjective
	}
	setInt := func(name string, dst *int, v int, skipZero bool) {
		if set[name] {
			return
		}
		if skipZero && v == 0 {
			return
		}
		*dst = v
	}
	setF64 := func(name string, dst *float64, v float64) {
		if !set[name] {
			*dst = v
		}
	}
	setStr := func(name string, dst *string, v string) {
		if !set[name] && v != "" {
			*dst = v
		}
	}
	setInt("rounds", rounds, p.Rounds, true)
	setInt("depth", depth, p.Depth, true)
	setInt("max-leaves", maxLeaves, p.MaxLeaves, false) // 0 有意义
	setF64("lr", lr, p.LR)
	setF64("lambda", lambda, p.Lambda)
	setF64("min-child-weight", minChildWeight, p.MinChildWeight)
	setF64("gamma", gamma, p.Gamma)
	setInt("max-bin", maxBin, p.MaxBin, true)
	setF64("subsample", subsample, p.Subsample)
	setF64("colsample", colsample, p.Colsample)
	setStr("tree-method", treeMethod, p.TreeMethod)
	if !set["seed"] && p.Seed != 0 {
		*seed = p.Seed
	}
	setInt("num-class", numClass, p.NumClass, true)
	setInt("ndcg-k", ndcgK, p.NDCGK, true)
	setInt("early-stop", earlyStop, p.EarlyStop, true)
	setInt("cv", cv, p.CVFolds, true)
	if !set["eval-metric"] && p.EvalMetric != "" {
		*evalMetric = p.EvalMetric
	}
}
