package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"math/rand"
	"os"
	"sort"
	"strconv"
	"strings"
)

func cmdRuns(args []string) error {
	sub := "status"
	rest := args
	if len(args) > 0 && !strings.HasPrefix(args[0], "-") {
		sub = strings.ToLower(strings.TrimSpace(args[0]))
		rest = args[1:]
	}
	switch sub {
	case "status", "summary":
		return cmdRunsStatus(rest)
	case "pareto":
		return cmdRunsPareto(rest)
	case "select-parent", "select_parent", "parent":
		return cmdRunsSelectParent(rest)
	case "check-dup", "check_dup", "dup":
		return cmdRunsCheckDup(rest)
	case "crossover", "merge":
		return cmdRunsCrossover(rest)
	case "propose", "mutate":
		return cmdRunsPropose(rest)
	default:
		return errUsage(fmt.Sprintf("未知 runs 子命令 %q（支持 status|pareto|select-parent|check-dup|crossover|propose）", sub))
	}
}

func readAllRuns(path string) ([]runRecord, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, errAgentWrap("data_load", fmt.Sprintf("open runs %s: %v", path, err),
			"确认 runs.jsonl 路径存在且由 train --runs 写出", false, err)
	}
	defer f.Close()
	var out []runRecord
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	lineNo := 0
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		lineNo++
		var rec runRecord
		if err := json.Unmarshal([]byte(line), &rec); err != nil {
			return nil, errAgent("data_load", fmt.Sprintf("runs.jsonl 第 %d 行 JSON 无效: %v", lineNo, err),
				"账本须由 leaves train --runs 追加", false)
		}
		out = append(out, rec)
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}
	return out, nil
}

func cmdRunsStatus(args []string) error {
	fs := flag.NewFlagSet("runs status", flag.ContinueOnError)
	runsPath := fs.String("runs", "", "runs.jsonl 路径（必需）")
	metricsPath := fs.String("metrics", "", "输出 JSON 路径（空=stdout）")
	tol := fs.Float64("rel-tol", 0.001, "相对改进收敛阈值（默认 0.001 即 0.1%）")
	patience := fs.Int("patience", 2, "连续无显著改进轮数判停阈值（默认 2）")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *runsPath == "" {
		return errUsage("leaves runs status 需要 --runs PATH")
	}
	records, err := readAllRuns(*runsPath)
	if err != nil {
		return err
	}
	if len(records) == 0 {
		return writeJSON(*metricsPath, map[string]any{
			"schema_version": metricsSchemaVersion,
			"total_runs":     0,
			"converged":      false,
		})
	}

	var best runRecord
	best = records[0]
	noImprove := 0
	var relChanges []float64
	bestSoFar := records[0].Value

	depths := map[int]int{}
	lrs := map[float64]int{}
	lambdas := map[float64]int{}
	subsamples := map[float64]int{}

	for i, r := range records {
		if r.Params != nil {
			depths[r.Params.Depth]++
			lrs[r.Params.LR]++
			lambdas[r.Params.Lambda]++
			subsamples[r.Params.Subsample]++
		}
		if i == 0 {
			continue
		}
		improved := false
		var rel float64
		denom := math.Abs(bestSoFar)
		if denom < 1e-12 {
			denom = 1.0
		}
		if r.Maximize {
			rel = (r.Value - bestSoFar) / denom
			if r.Value > bestSoFar {
				bestSoFar = r.Value
				best = r
				improved = rel >= *tol
			}
		} else {
			rel = (bestSoFar - r.Value) / denom
			if r.Value < bestSoFar {
				bestSoFar = r.Value
				best = r
				improved = rel >= *tol
			}
		}
		relChanges = append(relChanges, rel)
		if improved {
			noImprove = 0
		} else {
			noImprove++
		}
	}

	converged := len(records) >= (*patience+1) && noImprove >= *patience
	reason := "in_progress"
	if converged {
		reason = fmt.Sprintf("consecutive_no_significant_improve>=%d (rel_tol=%g)", *patience, *tol)
	}

	doc := map[string]any{
		"schema_version":          metricsSchemaVersion,
		"total_runs":              len(records),
		"best_tag":                best.Tag,
		"best_value":              best.Value,
		"metric":                  best.Metric,
		"maximize":                best.Maximize,
		"best_run":                best,
		"consecutive_no_improve":  noImprove,
		"recent_rel_improvements": relChanges,
		"converged":               converged,
		"convergence_reason":      reason,
		"explored_axes": map[string]int{
			"distinct_depths":     len(depths),
			"distinct_lrs":        len(lrs),
			"distinct_lambdas":    len(lambdas),
			"distinct_subsamples": len(subsamples),
		},
	}
	return writeJSON(*metricsPath, doc)
}

func cmdRunsPareto(args []string) error {
	fs := flag.NewFlagSet("runs pareto", flag.ContinueOnError)
	runsPath := fs.String("runs", "", "runs.jsonl 路径（必需）")
	metricsPath := fs.String("metrics", "", "输出 JSON 路径（空=stdout）")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *runsPath == "" {
		return errUsage("leaves runs pareto 需要 --runs PATH")
	}
	records, err := readAllRuns(*runsPath)
	if err != nil {
		return err
	}
	foldFront := computeFoldPareto(records)
	costFront := computeCostPareto(records)

	foldTags := make([]string, len(foldFront))
	for i, r := range foldFront {
		foldTags[i] = r.Tag
	}
	costTags := make([]string, len(costFront))
	for i, r := range costFront {
		costTags[i] = r.Tag
	}
	return writeJSON(*metricsPath, map[string]any{
		"schema_version":   metricsSchemaVersion,
		"total_runs":       len(records),
		"fold_pareto_tags": foldTags,
		"fold_pareto":      foldFront,
		"cost_pareto_tags": costTags,
		"cost_pareto":      costFront,
	})
}

func cmdRunsSelectParent(args []string) error {
	fs := flag.NewFlagSet("runs select-parent", flag.ContinueOnError)
	runsPath := fs.String("runs", "", "runs.jsonl 路径（必需）")
	strategy := fs.String("strategy", "pareto", "选父策略：pareto（默认）| top-k")
	seed := fs.Int64("seed", 42, "随机种子（确定性加权抽样）")
	metricsPath := fs.String("metrics", "", "输出 JSON 路径（空=stdout）")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *runsPath == "" {
		return errUsage("leaves runs select-parent 需要 --runs PATH")
	}
	records, err := readAllRuns(*runsPath)
	if err != nil {
		return err
	}
	if len(records) == 0 {
		return errAgent("data_load", "runs.jsonl 为空，无法选父代", "先执行至少一次 leaves train --runs", false)
	}

	var front []runRecord
	if strings.EqualFold(strings.TrimSpace(*strategy), "top-k") {
		sorted := append([]runRecord(nil), records...)
		sortRunsByMetric(sorted)
		k := 3
		if len(sorted) < k {
			k = len(sorted)
		}
		front = sorted[:k]
	} else {
		front = computeFoldPareto(records)
		if len(front) == 0 {
			front = computeCostPareto(records)
		}
	}
	// 计算每个前沿候选在各折上夺冠的频次作为权重（基础权重为 1）
	weights := make([]int, len(front))
	for i := range weights {
		weights[i] = 1
	}
	if len(front) > 1 && len(front[0].FoldMetrics) > 0 {
		k := len(front[0].FoldMetrics)
		for f := 0; f < k; f++ {
			bestIdx := 0
			for i := 1; i < len(front); i++ {
				if f >= len(front[i].FoldMetrics) {
					continue
				}
				if front[i].Maximize {
					if front[i].FoldMetrics[f] > front[bestIdx].FoldMetrics[f] {
						bestIdx = i
					}
				} else {
					if front[i].FoldMetrics[f] < front[bestIdx].FoldMetrics[f] {
						bestIdx = i
					}
				}
			}
			weights[bestIdx] += 2
		}
	}

	totalW := 0
	for _, w := range weights {
		totalW += w
	}
	rng := rand.New(rand.NewSource(*seed + int64(len(records))))
	pick := rng.Intn(totalW)
	chosenIdx := 0
	acc := 0
	for i, w := range weights {
		acc += w
		if pick < acc {
			chosenIdx = i
			break
		}
	}
	chosen := front[chosenIdx]
	suggAxes := suggestNextMutationAxes(records)
	candTags := make([]string, len(front))
	for i, c := range front {
		candTags[i] = c.Tag
	}

	return writeJSON(*metricsPath, map[string]any{
		"schema_version":   metricsSchemaVersion,
		"selected_tag":     chosen.Tag,
		"selected_value":   chosen.Value,
		"selected_run":     chosen,
		"pareto_pool_tags": candTags,
		"suggested_axes":   suggAxes,
	})
}

func cmdRunsCheckDup(args []string) error {
	fs := flag.NewFlagSet("runs check-dup", flag.ContinueOnError)
	runsPath := fs.String("runs", "", "runs.jsonl 路径（必需）")
	paramsJSON := fs.String("params", "", "候选超参 JSON 对象（如 '{\"depth\":6,\"lr\":0.1}'）")
	metricsPath := fs.String("metrics", "", "输出 JSON 路径（空=stdout）")
	rounds := fs.Int("rounds", 0, "候选 rounds")
	depth := fs.Int("depth", 0, "候选 depth")
	maxLeaves := fs.Int("max-leaves", -1, "候选 max-leaves")
	lr := fs.Float64("lr", 0, "候选 lr")
	lambda := fs.Float64("lambda", -1, "候选 lambda")
	minChildWeight := fs.Float64("min-child-weight", -1, "候选 min-child-weight")
	gamma := fs.Float64("gamma", -1, "候选 gamma")
	subsample := fs.Float64("subsample", 0, "候选 subsample")
	colsample := fs.Float64("colsample", 0, "候选 colsample")
	boosterName := fs.String("booster", "", "候选 booster")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *runsPath == "" {
		return errUsage("leaves runs check-dup 需要 --runs PATH")
	}
	records, err := readAllRuns(*runsPath)
	if err != nil {
		return err
	}
	set := map[string]bool{}
	fs.Visit(func(f *flag.Flag) { set[f.Name] = true })

	if strings.TrimSpace(*paramsJSON) != "" {
		var pm map[string]any
		if err := json.Unmarshal([]byte(*paramsJSON), &pm); err != nil {
			return errUsage(fmt.Sprintf("--params JSON 解析失败: %v", err))
		}
		if v, ok := asFloat(pm["rounds"]); ok {
			*rounds = int(v)
			set["rounds"] = true
		}
		if v, ok := asFloat(pm["depth"]); ok {
			*depth = int(v)
			set["depth"] = true
		}
		if v, ok := asFloat(pm["max_leaves"]); ok {
			*maxLeaves = int(v)
			set["max-leaves"] = true
		}
		if v, ok := asFloat(pm["lr"]); ok {
			*lr = v
			set["lr"] = true
		}
		if v, ok := asFloat(pm["lambda"]); ok {
			*lambda = v
			set["lambda"] = true
		}
		if v, ok := asFloat(pm["min_child_weight"]); ok {
			*minChildWeight = v
			set["min-child-weight"] = true
		}
		if v, ok := asFloat(pm["gamma"]); ok {
			*gamma = v
			set["gamma"] = true
		}
		if v, ok := asFloat(pm["subsample"]); ok {
			*subsample = v
			set["subsample"] = true
		}
		if v, ok := asFloat(pm["colsample"]); ok {
			*colsample = v
			set["colsample"] = true
		}
		if s, ok := pm["booster"].(string); ok && s != "" {
			*boosterName = s
			set["booster"] = true
		}
	}

	var matched *runRecord
	for i := range records {
		p := records[i].Params
		if p == nil {
			continue
		}
		if set["rounds"] && p.Rounds != *rounds {
			continue
		}
		if set["depth"] && p.Depth != *depth {
			continue
		}
		if set["max-leaves"] && p.MaxLeaves != *maxLeaves {
			continue
		}
		if set["lr"] && math.Abs(p.LR-*lr) > 1e-9 {
			continue
		}
		if set["lambda"] && math.Abs(p.Lambda-*lambda) > 1e-9 {
			continue
		}
		if set["min-child-weight"] && math.Abs(p.MinChildWeight-*minChildWeight) > 1e-9 {
			continue
		}
		if set["gamma"] && math.Abs(p.Gamma-*gamma) > 1e-9 {
			continue
		}
		if set["subsample"] && math.Abs(p.Subsample-*subsample) > 1e-9 {
			continue
		}
		if set["colsample"] && math.Abs(p.Colsample-*colsample) > 1e-9 {
			continue
		}
		if set["booster"] {
			b := p.Booster
			if b == "" {
				b = "gbtree"
			}
			if !strings.EqualFold(b, *boosterName) {
				continue
			}
		}
		matched = &records[i]
		break
	}

	res := map[string]any{
		"schema_version": metricsSchemaVersion,
		"duplicate":      matched != nil,
	}
	if matched != nil {
		res["matched_tag"] = matched.Tag
		res["matched_value"] = matched.Value
	}
	return writeJSON(*metricsPath, res)
}

// cmdRunsCrossover 实现 SKILL §4.5 ③ 交叉重组算子（Merge / Crossover，WP-36）：
// 从两个前沿父代 A、B 各取半参（A 的结构/步长参数 + B 的正则/采样参数），并保证不与账本历史重复。
func cmdRunsCrossover(args []string) error {
	fs := flag.NewFlagSet("runs crossover", flag.ContinueOnError)
	runsPath := fs.String("runs", "", "runs.jsonl 路径（必需）")
	parentATag := fs.String("parent-a", "", "父代 A 标签（结构/步长提供方；空=自动从前沿选取）")
	parentBTag := fs.String("parent-b", "", "父代 B 标签（正则/采样提供方；空=自动从前沿选取）")
	seed := fs.Int64("seed", 42, "随机种子")
	metricsPath := fs.String("metrics", "", "输出 JSON 路径（空=stdout）")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *runsPath == "" {
		return errUsage("leaves runs crossover 需要 --runs PATH")
	}
	records, err := readAllRuns(*runsPath)
	if err != nil {
		return err
	}
	if len(records) == 0 {
		return errAgent("data_load", "runs.jsonl 为空，无法执行交叉重组", "先执行至少两次 leaves train --runs", false)
	}

	pool := buildParentPool(records)
	var parentA, parentB runRecord
	foundA, foundB := false, false
	if *parentATag != "" {
		for _, r := range records {
			if r.Tag == *parentATag {
				parentA = r
				foundA = true
			}
		}
		if !foundA {
			return errUsage(fmt.Sprintf("--parent-a %q 不在账本中", *parentATag))
		}
	}
	if *parentBTag != "" {
		for _, r := range records {
			if r.Tag == *parentBTag {
				parentB = r
				foundB = true
			}
		}
		if !foundB {
			return errUsage(fmt.Sprintf("--parent-b %q 不在账本中", *parentBTag))
		}
	}

	rng := rand.New(rand.NewSource(*seed + int64(len(records))))
	if !foundA {
		parentA = pool[0]
		if len(pool) > 1 && rng.Float64() < 0.3 {
			parentA = pool[rng.Intn(len(pool))]
		}
	}
	if !foundB {
		for _, cand := range pool {
			if cand.Tag != parentA.Tag {
				parentB = cand
				foundB = true
				break
			}
		}
		if !foundB {
			for _, cand := range records {
				if cand.Tag != parentA.Tag {
					parentB = cand
					foundB = true
					break
				}
			}
		}
		if !foundB {
			parentB = parentA
		}
	}

	pa := cloneParamsOrDefault(parentA.Params)
	pb := cloneParamsOrDefault(parentB.Params)

	// A 提供结构与步长旋钮，B 提供正则与随机采样旋钮
	child := cloneParamsOrDefault(&pa)
	child.Lambda = pb.Lambda
	child.MinChildWeight = pb.MinChildWeight
	child.Gamma = pb.Gamma
	child.Subsample = pb.Subsample
	child.Colsample = pb.Colsample
	if pb.MaxBin > 0 {
		child.MaxBin = pb.MaxBin
	}

	ensureNonDuplicateParams(&child, records)

	tagA := parentA.Tag
	if tagA == "" {
		tagA = "pA"
	}
	tagB := parentB.Tag
	if tagB == "" {
		tagB = "pB"
	}
	suggTag := fmt.Sprintf("x:%s|%s", tagA, tagB)
	obj := parentA.Objective
	if obj == "" {
		obj = parentB.Objective
	}

	return writeJSON(*metricsPath, map[string]any{
		"schema_version":      metricsSchemaVersion,
		"parent_a":            tagA,
		"parent_b":            tagB,
		"suggested_tag":       suggTag,
		"objective":           obj,
		"params":              child,
		"duplicate_in_ledger": isDuplicateInLedger(&child, records),
		"suggested_cli_args":  paramsToCLIArgs(obj, &child, suggTag),
	})
}

// cmdRunsPropose 实现 SKILL §4.5 ② 反射式定向变异提议算子（WP-36）：
// 结合父代与可选 diagnose 诊断结论，自动生成保证不在账本中重复的下一代超参、谱系 tag 与 CLI 参数。
func cmdRunsPropose(args []string) error {
	fs := flag.NewFlagSet("runs propose", flag.ContinueOnError)
	runsPath := fs.String("runs", "", "runs.jsonl 路径（必需）")
	parentTag := fs.String("parent", "", "指定父代 tag（空=自动从 Pareto 前沿选优）")
	diagPath := fs.String("diagnose", "", "可选：leaves diagnose 产出的诊断 JSON 路径")
	seed := fs.Int64("seed", 42, "随机种子")
	metricsPath := fs.String("metrics", "", "输出 JSON 路径（空=stdout）")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *runsPath == "" {
		return errUsage("leaves runs propose 需要 --runs PATH")
	}
	records, err := readAllRuns(*runsPath)
	if err != nil {
		return err
	}
	if len(records) == 0 {
		return errAgent("data_load", "runs.jsonl 为空，无法提议变异", "先执行至少一次 leaves train --runs", false)
	}

	var parent runRecord
	found := false
	if *parentTag != "" {
		for _, r := range records {
			if r.Tag == *parentTag {
				parent = r
				found = true
			}
		}
		if !found {
			return errUsage(fmt.Sprintf("--parent %q 不在账本中", *parentTag))
		}
	} else {
		pool := buildParentPool(records)
		rng := rand.New(rand.NewSource(*seed + int64(len(records))))
		parent = pool[rng.Intn(len(pool))]
	}

	diagnosis := ""
	var diagDropCols []string
	if *diagPath != "" {
		b, err := os.ReadFile(*diagPath)
		if err != nil {
			return errAgentWrap("data_load", fmt.Sprintf("read --diagnose: %v", err), "检查诊断文件路径", false, err)
		}
		var ddoc map[string]any
		if err := json.Unmarshal(b, &ddoc); err != nil {
			return errAgentWrap("data_load", fmt.Sprintf("parse --diagnose: %v", err), "确认诊断文件 JSON 有效", false, err)
		}
		if d, ok := ddoc["diagnosis"].(string); ok {
			diagnosis = d
		}
		if sdc, ok := ddoc["suggested_drop_cols"].([]any); ok {
			for _, x := range sdc {
				if s, ok := x.(string); ok && s != "" {
					diagDropCols = append(diagDropCols, s)
				}
			}
		}
		if spw, ok := asFloat(ddoc["suggested_scale_pos_weight"]); ok && spw > 0 {
			if parent.Params == nil || parent.Params.ScalePosWeight <= 0 {
				p := cloneParamsOrDefault(parent.Params)
				p.ScalePosWeight = spw
				parent.Params = &p
			}
		}
	}

	base := cloneParamsOrDefault(parent.Params)
	if len(diagDropCols) > 0 {
		merged := mergeDropCols(base.DropCols, diagDropCols)
		base.DropCols = merged
	}

	mutated, mutSummary := proposeNonDuplicateMutation(base, diagnosis, records)
	pTag := parent.Tag
	if pTag == "" {
		pTag = "baseline"
	}
	suggTag := fmt.Sprintf("p:%s+%s", pTag, mutSummary)
	obj := parent.Objective

	return writeJSON(*metricsPath, map[string]any{
		"schema_version":      metricsSchemaVersion,
		"parent_tag":          pTag,
		"parent_value":        parent.Value,
		"diagnosis_used":      diagnosis,
		"mutation_summary":    mutSummary,
		"suggested_tag":       suggTag,
		"objective":           obj,
		"params":              mutated,
		"duplicate_in_ledger": isDuplicateInLedger(&mutated, records),
		"suggested_cli_args":  paramsToCLIArgs(obj, &mutated, suggTag),
	})
}

func buildParentPool(records []runRecord) []runRecord {
	front := computeFoldPareto(records)
	seen := map[string]bool{}
	var pool []runRecord
	for _, r := range front {
		seen[r.Tag] = true
		pool = append(pool, r)
	}
	if len(pool) < 2 {
		for _, r := range computeCostPareto(records) {
			if !seen[r.Tag] {
				seen[r.Tag] = true
				pool = append(pool, r)
			}
		}
	}
	if len(pool) < 2 {
		sorted := append([]runRecord(nil), records...)
		sortRunsByMetric(sorted)
		for _, r := range sorted {
			if !seen[r.Tag] {
				seen[r.Tag] = true
				pool = append(pool, r)
			}
			if len(pool) >= 3 {
				break
			}
		}
	}
	if len(pool) == 0 && len(records) > 0 {
		pool = []runRecord{records[0]}
	}
	return pool
}

func sortRunsByMetric(rs []runRecord) {
	sort.SliceStable(rs, func(i, j int) bool {
		return betterRun(&rs[i], &rs[j])
	})
}

func cloneParamsOrDefault(p *paramsRecord) paramsRecord {
	if p == nil {
		return paramsRecord{
			Rounds:         50,
			Depth:          6,
			LR:             0.3,
			Lambda:         1.0,
			MinChildWeight: 1.0,
			MaxBin:         256,
			Subsample:      1.0,
			Colsample:      1.0,
			TreeMethod:     "auto",
			Seed:           42,
		}
	}
	out := *p
	if out.Rounds <= 0 {
		out.Rounds = 50
	}
	if out.Depth <= 0 {
		out.Depth = 6
	}
	if out.LR <= 0 {
		out.LR = 0.3
	}
	if out.MaxBin <= 0 {
		out.MaxBin = 256
	}
	if out.Subsample <= 0 {
		out.Subsample = 1.0
	}
	if out.Colsample <= 0 {
		out.Colsample = 1.0
	}
	if out.TreeMethod == "" {
		out.TreeMethod = "auto"
	}
	return out
}

func paramsEqual(a, b *paramsRecord) bool {
	if a == nil || b == nil {
		return false
	}
	return a.Rounds == b.Rounds &&
		a.Depth == b.Depth &&
		a.MaxLeaves == b.MaxLeaves &&
		math.Abs(a.LR-b.LR) <= 1e-9 &&
		math.Abs(a.Lambda-b.Lambda) <= 1e-9 &&
		math.Abs(a.MinChildWeight-b.MinChildWeight) <= 1e-9 &&
		math.Abs(a.Gamma-b.Gamma) <= 1e-9 &&
		math.Abs(a.Subsample-b.Subsample) <= 1e-9 &&
		math.Abs(a.Colsample-b.Colsample) <= 1e-9 &&
		a.DropCols == b.DropCols &&
		strings.EqualFold(defaultBooster(a.Booster), defaultBooster(b.Booster))
}

func defaultBooster(b string) string {
	if b == "" {
		return "gbtree"
	}
	return b
}

func isDuplicateInLedger(cand *paramsRecord, records []runRecord) bool {
	for i := range records {
		if paramsEqual(cand, records[i].Params) {
			return true
		}
	}
	return false
}

func ensureNonDuplicateParams(cand *paramsRecord, records []runRecord) {
	if !isDuplicateInLedger(cand, records) {
		return
	}
	lambdas := []float64{2.0, 3.0, 5.0, 10.0, 0.5, 1.5}
	for _, l := range lambdas {
		trial := *cand
		trial.Lambda = l
		if !isDuplicateInLedger(&trial, records) {
			*cand = trial
			return
		}
	}
	depths := []int{4, 5, 7, 8, 3}
	for _, d := range depths {
		trial := *cand
		trial.Depth = d
		if !isDuplicateInLedger(&trial, records) {
			*cand = trial
			return
		}
	}
	cand.Rounds += 25
}

type candidateMutation struct {
	summary string
	apply   func(p *paramsRecord)
}

func proposeNonDuplicateMutation(base paramsRecord, diagnosis string, records []runRecord) (paramsRecord, string) {
	var cands []candidateMutation
	switch diagnosis {
	case "overfitting":
		cands = append(cands,
			candidateMutation{"depth_down_lambda2", func(p *paramsRecord) {
				if p.Depth > 3 {
					p.Depth--
				}
				p.Lambda = math.Max(2.0, p.Lambda*2.0)
			}},
			candidateMutation{"subsample08_colsample08", func(p *paramsRecord) {
				p.Subsample = 0.8
				p.Colsample = 0.8
			}},
			candidateMutation{"mcw5_gamma1", func(p *paramsRecord) {
				p.MinChildWeight = math.Max(5.0, p.MinChildWeight*2.5)
				p.Gamma = 1.0
			}},
		)
	case "underfitting":
		cands = append(cands,
			candidateMutation{"depth_up_rounds2x", func(p *paramsRecord) {
				p.Depth += 2
				if p.Depth > 10 {
					p.Depth = 10
				}
				p.Rounds *= 2
			}},
			candidateMutation{"lr01_rounds3x", func(p *paramsRecord) {
				p.LR = 0.1
				p.Rounds *= 3
			}},
		)
	case "oscillating":
		cands = append(cands,
			candidateMutation{"lr_half_rounds2x", func(p *paramsRecord) {
				p.LR = math.Max(0.01, p.LR*0.5)
				p.Rounds *= 2
			}},
			candidateMutation{"mcw3_lambda3", func(p *paramsRecord) {
				p.MinChildWeight = 3.0
				p.Lambda = 3.0
			}},
		)
	case "high_variance":
		cands = append(cands,
			candidateMutation{"shallow_mcw5", func(p *paramsRecord) {
				if p.Depth > 4 {
					p.Depth = 4
				} else if p.Depth > 2 {
					p.Depth--
				}
				p.MinChildWeight = 5.0
				p.Lambda = 3.0
			}},
		)
	case "feature_noise":
		cands = append(cands,
			candidateMutation{"prune_cols_colsample08", func(p *paramsRecord) {
				p.Colsample = 0.8
			}},
		)
	}

	// 通用探索网格兜底（按 SKILL §4.4 顺序）
	cands = append(cands,
		candidateMutation{"depth4_lr01_r200", func(p *paramsRecord) { p.Depth = 4; p.LR = 0.1; p.Rounds = 200 }},
		candidateMutation{"depth8_lr005_r300", func(p *paramsRecord) { p.Depth = 8; p.LR = 0.05; p.Rounds = 300 }},
		candidateMutation{"lambda3_mcw5", func(p *paramsRecord) { p.Lambda = 3.0; p.MinChildWeight = 5.0 }},
		candidateMutation{"sub08_col08", func(p *paramsRecord) { p.Subsample = 0.8; p.Colsample = 0.8 }},
		candidateMutation{"gamma1_lambda5", func(p *paramsRecord) { p.Gamma = 1.0; p.Lambda = 5.0 }},
		candidateMutation{"booster_dart", func(p *paramsRecord) { p.Booster = "dart"; p.RateDrop = 0.1 }},
	)

	for _, c := range cands {
		trial := base
		c.apply(&trial)
		if !isDuplicateInLedger(&trial, records) {
			return trial, c.summary
		}
	}
	fallback := base
	ensureNonDuplicateParams(&fallback, records)
	return fallback, fmt.Sprintf("r%d_d%d_l%g", fallback.Rounds, fallback.Depth, fallback.Lambda)
}

func mergeDropCols(existing string, extra []string) string {
	seen := map[string]bool{}
	var out []string
	for _, s := range strings.Split(existing, ",") {
		s = strings.TrimSpace(s)
		if s != "" && !seen[s] {
			seen[s] = true
			out = append(out, s)
		}
	}
	for _, s := range extra {
		s = strings.TrimSpace(s)
		if s != "" && !seen[s] {
			seen[s] = true
			out = append(out, s)
		}
	}
	return strings.Join(out, ",")
}

func paramsToCLIArgs(obj string, p *paramsRecord, tag string) []string {
	var args []string
	if obj != "" {
		args = append(args, "--objective", obj)
	}
	if p == nil {
		return args
	}
	args = append(args,
		"--rounds", strconv.Itoa(p.Rounds),
		"--depth", strconv.Itoa(p.Depth),
		"--lr", strconv.FormatFloat(p.LR, 'f', -1, 64),
		"--lambda", strconv.FormatFloat(p.Lambda, 'f', -1, 64),
		"--min-child-weight", strconv.FormatFloat(p.MinChildWeight, 'f', -1, 64),
	)
	if p.Gamma > 0 {
		args = append(args, "--gamma", strconv.FormatFloat(p.Gamma, 'f', -1, 64))
	}
	if p.Subsample > 0 && p.Subsample < 1.0 {
		args = append(args, "--subsample", strconv.FormatFloat(p.Subsample, 'f', -1, 64))
	}
	if p.Colsample > 0 && p.Colsample < 1.0 {
		args = append(args, "--colsample", strconv.FormatFloat(p.Colsample, 'f', -1, 64))
	}
	if p.MaxLeaves > 0 {
		args = append(args, "--max-leaves", strconv.Itoa(p.MaxLeaves))
	}
	if p.CVFolds >= 2 {
		args = append(args, "--cv", strconv.Itoa(p.CVFolds))
	}
	if p.EarlyStop > 0 {
		args = append(args, "--early-stop", strconv.Itoa(p.EarlyStop))
	}
	if p.Booster != "" && p.Booster != "gbtree" {
		args = append(args, "--booster", p.Booster)
	}
	if p.LabelCol != "" {
		args = append(args, "--label-col", p.LabelCol)
	}
	if p.DropCols != "" {
		args = append(args, "--drop-cols", p.DropCols)
	}
	if p.NAPolicy != "" && p.NAPolicy != "error" {
		args = append(args, "--na-policy", p.NAPolicy)
	}
	if p.AutoEncode {
		args = append(args, "--auto-encode")
	}
	if p.NumClass >= 2 {
		args = append(args, "--num-class", strconv.Itoa(p.NumClass))
	}
	if p.NumTarget >= 2 {
		args = append(args, "--num-target", strconv.Itoa(p.NumTarget))
	}
	if p.EvalMetric != "" {
		args = append(args, "--eval-metric", p.EvalMetric)
	}
	if p.WeightCol != "" {
		args = append(args, "--weight-col", p.WeightCol)
	}
	if p.ScalePosWeight > 0 {
		args = append(args, "--scale-pos-weight", strconv.FormatFloat(p.ScalePosWeight, 'f', -1, 64))
	}
	if p.TweedieVariancePower > 0 {
		args = append(args, "--tweedie-variance-power", strconv.FormatFloat(p.TweedieVariancePower, 'f', -1, 64))
	}
	if p.MultiOutputTree {
		args = append(args, "--multi-output-tree")
	}
	if p.NumParallelTree > 0 {
		args = append(args, "--num-parallel-tree", strconv.Itoa(p.NumParallelTree))
	}
	if tag != "" {
		args = append(args, "--tag", tag)
	}
	return args
}

// computeFoldPareto 计算折级 Pareto 前沿：若存在 fold_metrics，寻找未被任何其他行在所有折上支配的候选。
func computeFoldPareto(records []runRecord) []runRecord {
	var withFolds []runRecord
	for _, r := range records {
		if len(r.FoldMetrics) > 0 {
			withFolds = append(withFolds, r)
		}
	}
	if len(withFolds) == 0 {
		// 回退为标量最优单点
		if len(records) == 0 {
			return nil
		}
		best := records[0]
		for _, r := range records[1:] {
			if betterRun(&r, &best) {
				best = r
			}
		}
		return []runRecord{best}
	}
	var front []runRecord
	for i, a := range withFolds {
		dominated := false
		for j, b := range withFolds {
			if i == j {
				continue
			}
			if dominatesAcrossFolds(b, a) {
				dominated = true
				break
			}
		}
		if !dominated {
			front = append(front, a)
		}
	}
	return front
}

func dominatesAcrossFolds(b, a runRecord) bool {
	if len(b.FoldMetrics) == 0 || len(b.FoldMetrics) != len(a.FoldMetrics) {
		return false
	}
	atLeastOneStrictlyBetter := false
	for k := range b.FoldMetrics {
		bv, av := b.FoldMetrics[k], a.FoldMetrics[k]
		if b.Maximize {
			if bv < av {
				return false
			}
			if bv > av {
				atLeastOneStrictlyBetter = true
			}
		} else {
			if bv > av {
				return false
			}
			if bv < av {
				atLeastOneStrictlyBetter = true
			}
		}
	}
	return atLeastOneStrictlyBetter
}

// computeCostPareto 计算 (value, n_trees) 双目标非支配前沿（同指标或更优指标且树数更少）。
func computeCostPareto(records []runRecord) []runRecord {
	var front []runRecord
	for i, a := range records {
		dominated := false
		for j, b := range records {
			if i == j {
				continue
			}
			if dominatesOnValueAndCost(b, a) {
				dominated = true
				break
			}
		}
		if !dominated {
			front = append(front, a)
		}
	}
	return front
}

func dominatesOnValueAndCost(b, a runRecord) bool {
	valBetterOrEq := false
	valStrict := false
	if b.Maximize {
		valBetterOrEq = b.Value >= a.Value
		valStrict = b.Value > a.Value
	} else {
		valBetterOrEq = b.Value <= a.Value
		valStrict = b.Value < a.Value
	}
	if !valBetterOrEq {
		return false
	}
	aTrees := a.NTrees
	bTrees := b.NTrees
	if aTrees <= 0 {
		aTrees = math.MaxInt32
	}
	if bTrees <= 0 {
		bTrees = math.MaxInt32
	}
	costBetterOrEq := bTrees <= aTrees
	costStrict := bTrees < aTrees
	if !costBetterOrEq {
		return false
	}
	return valStrict || costStrict
}

func suggestNextMutationAxes(records []runRecord) []string {
	depths := map[int]bool{}
	lambdas := map[float64]bool{}
	subsamples := map[float64]bool{}
	lrs := map[float64]bool{}
	for _, r := range records {
		if r.Params == nil {
			continue
		}
		depths[r.Params.Depth] = true
		lambdas[r.Params.Lambda] = true
		subsamples[r.Params.Subsample] = true
		lrs[r.Params.LR] = true
	}
	var out []string
	if len(depths) <= 1 {
		out = append(out, "depth (structure)")
	}
	if len(lambdas) <= 1 {
		out = append(out, "lambda / min-child-weight (regularization)")
	}
	if len(subsamples) <= 1 {
		out = append(out, "subsample / colsample (stochastic)")
	}
	if len(lrs) <= 1 {
		out = append(out, "lr + rounds (step_size)")
	}
	if len(out) == 0 {
		out = []string{"fine_tune_around_parent", "booster=dart"}
	}
	return out
}
