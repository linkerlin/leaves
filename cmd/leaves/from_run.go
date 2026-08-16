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
//   - tag 非空但账本中不存在：回落最优行（notice 说明），保留用户新 tag——
//     支持演化谱系流程「--from-run 复现最优 + --tag p:parent+mutation 起新名」
func loadRunFromLedger(path, tag string) (*runRecord, string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, "", errAgentWrap("data_load", fmt.Sprintf("open --from-run: %v", err),
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
			return nil, "", errAgent("data_load", fmt.Sprintf("runs.jsonl 第 %d 行 JSON 无效: %v", n+1, err),
				"账本须由 leaves train --runs 追加，勿手写损坏行", false)
		}
		n++
		cp := rec
		if !bestInit {
			best = &cp
			bestInit = true
		} else if betterRun(&cp, best) {
			best = &cp
		}
		if tag != "" {
			if rec.Tag == tag {
				cp2 := rec
				byTag = &cp2
			}
		}
	}
	if err := sc.Err(); err != nil {
		return nil, "", fmt.Errorf("read runs: %w", err)
	}
	if n == 0 {
		return nil, "", errAgent("data_load", "runs.jsonl 为空",
			"先 train --runs PATH --tag NAME 写入至少一行", false)
	}
	if tag != "" {
		if byTag != nil {
			return byTag, "", nil
		}
		// 未命中：回落最优行。若与最优 tag 也不同名，仍可能是笔误——notice 让 Agent 可审计纠正。
		return best, fmt.Sprintf("runs.jsonl 中无 tag=%q，已回落最优行 tag=%q（若本意是复现某父代，请核对其 tag）", tag, best.Tag), nil
	}
	return best, "", nil
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
	numClass, numTarget, rounds, depth, maxLeaves *int,
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
	// 注意：val 不回填——--from-run 语义是「定稿/变异起点」（默认全量重训，见 SKILL 定稿），
	// 忠实重放某次早停 run 用 manifest.reproduce（其含 --val --early-stop）或显式传 --val。
	if !set["seed"] && p.Seed != 0 {
		*seed = p.Seed
	}
	setInt("num-class", numClass, p.NumClass, true)
	setInt("num-target", numTarget, p.NumTarget, true)
	setInt("ndcg-k", ndcgK, p.NDCGK, true)
	setInt("early-stop", earlyStop, p.EarlyStop, true)
	setInt("cv", cv, p.CVFolds, true)
	if !set["eval-metric"] && p.EvalMetric != "" {
		*evalMetric = p.EvalMetric
	}
}
