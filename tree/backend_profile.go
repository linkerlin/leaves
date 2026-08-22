package tree

import (
	"fmt"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"
)

// BackendTiming 单后端 profiling 计时结果。
type BackendTiming struct {
	Backend Backend
	// NsPerOp 每次 PredictDense 的纳秒（warm-up 后均值）。
	NsPerOp float64
	Ok      bool
}

// ProfileResult SelectBackend 第二轮：实际 workload 计时各后端并推荐最快。
type ProfileResult struct {
	Native  BackendTiming
	BornCPU BackendTiming
	BornGPU BackendTiming
	Pick    Backend
	Rule    string
	Reason  string
}

// ProfileBackend 对实际 workload 做短 profiling，测 Native / BornCPU / BornGPU 的 ns/op
// 并推荐最快可用后端。用于 BackendAuto 第二轮（数据驱动选型）。
//
// 不修改全局状态，不改 2.0 默认决策表（SelectBackendExplained）——本函数是 opt-in：
// 调用方显式传入代表性批量样本，得到测量证据后再决定 Backend。
//
// vals/nrows/ncols 为代表性批量输入（与真实预测同形）；iters 计时轮数（≤0 默认 20，建议 ≥10）。
// 不支持的后端（如 BornGPU 在无 WebGPU 环境）Ok=false 不参与推荐。
func ProfileBackend(caps ModelCaps, vals []float64, nrows, ncols, iters int) ProfileResult {
	res := ProfileResult{}
	if caps.Forest == nil || nrows <= 0 || ncols <= 0 || len(vals) < nrows*ncols {
		res.Pick = BackendNative
		res.Rule = "profile_invalid"
		res.Reason = "nil forest or invalid sample shape"
		return res
	}
	if iters <= 0 {
		iters = 20
	}
	groups := caps.Forest.NumOutputGroups
	if groups < 1 {
		groups = 1
	}
	transform := ApplyTransformRaw
	outType := TransformRaw

	// Native（始终可用）
	ne := NewNativeEngine(caps.Forest, transform, outType, groups)
	if ns, err := timePredictDense(ne, vals, nrows, ncols, groups, 0, iters); err == nil {
		res.Native = BackendTiming{Backend: BackendNative, NsPerOp: ns, Ok: true}
	}

	// BornCPU（数值树、无 cat-small）
	if BornSupports(caps.Forest, BackendBornCPU) {
		if be, err := NewBornEngine(caps.Forest, transform, outType, groups, &BornConfig{UseGPU: false}); err == nil {
			if ns, err := timePredictDense(be, vals, nrows, ncols, groups, 0, iters); err == nil {
				res.BornCPU = BackendTiming{Backend: BackendBornCPU, NsPerOp: ns, Ok: true}
			}
		}
	}

	// BornGPU（Windows + WebGPU 可用）
	if BornSupports(caps.Forest, BackendBornGPU) {
		if be, err := NewBornEngine(caps.Forest, transform, outType, groups, &BornConfig{UseGPU: true}); err == nil && be.BornUsingGPU() {
			if ns, err := timePredictDense(be, vals, nrows, ncols, groups, 0, iters); err == nil {
				res.BornGPU = BackendTiming{Backend: BackendBornGPU, NsPerOp: ns, Ok: true}
			}
		}
	}

	res.Pick, res.Rule, res.Reason = pickFastest(res.Native, res.BornCPU, res.BornGPU)
	return res
}

// timePredictDense warm-up 后计时 iters 次 PredictDense，返回 ns/op。
// nEstimators=0 表示全量树（各后端公平）。
func timePredictDense(eng Engine, vals []float64, nrows, ncols, groups, nEstimators, iters int) (float64, error) {
	pred := make([]float64, nrows*groups)
	warmup := 2
	if iters < warmup {
		warmup = iters
	}
	for i := 0; i < warmup; i++ {
		if err := eng.PredictDense(vals, nrows, ncols, pred, nEstimators); err != nil {
			return 0, err
		}
	}
	start := time.Now()
	for i := 0; i < iters; i++ {
		if err := eng.PredictDense(vals, nrows, ncols, pred, nEstimators); err != nil {
			return 0, err
		}
	}
	elapsed := time.Since(start).Seconds()
	return elapsed / float64(iters) * 1e9, nil
}

// pickFastest 选 Ok 中 NsPerOp 最小的；全不可用时回落 Native。
func pickFastest(timings ...BackendTiming) (pick Backend, rule, reason string) {
	var best BackendTiming
	for _, t := range timings {
		if !t.Ok {
			continue
		}
		if !best.Ok || t.NsPerOp < best.NsPerOp {
			best = t
		}
	}
	if !best.Ok {
		return BackendNative, "profile_none_ok", "no backend profiled ok → Native"
	}
	rule = "profile_" + BackendName(best.Backend)
	reason = fmt.Sprintf("profiling: %s fastest at %.0f ns/op (native=%.0f born_cpu=%.0f born_gpu=%.0f)",
		BackendName(best.Backend), best.NsPerOp,
		timings[0].NsPerOp, timings[1].NsPerOp, timings[2].NsPerOp)
	return best.Backend, rule, reason
}

// —— 二轮 profiling 的 Auto 接线（opt-in；BA-PROF）——
//
// LEAVES_BACKEND_PROFILE=1|on|true 时，SelectBackendExplained 在决策表之前
// 以合成批量样本实测各后端 ns/op，选最快（Rule=profile_*，Reason 携带数字）。
// 结果按 (batch, nfeatures, 森林规模) 形状类缓存在进程内——profiling 只发生
// 一次/形状类，不重复付首调用延迟。默认（未设 env）决策表 2.0 行为不变。

var profileCache sync.Map // string → BackendDecision

func backendProfilingEnabled() bool {
	switch strings.ToLower(strings.TrimSpace(os.Getenv("LEAVES_BACKEND_PROFILE"))) {
	case "1", "on", "true", "yes":
		return true
	}
	return false
}

// profileCacheKey 形状类键：批量档、特征数、森林规模（trees/nodes）。
func profileCacheKey(caps ModelCaps, hint WorkloadHint) string {
	batch := hint.BatchSize
	if batch <= 0 {
		batch = 1
	}
	f := caps.Forest
	nodes := 0
	for i := range f.Trees {
		nodes += f.Trees[i].NumNodes
	}
	return fmt.Sprintf("b=%d,f=%d,t=%d,n=%d", batch, f.NumFeatures, len(f.Trees), nodes)
}

// profiledDecision 查缓存或跑一次 ProfileBackend（合成确定性样本）。
func profiledDecision(caps ModelCaps, hint WorkloadHint) BackendDecision {
	key := profileCacheKey(caps, hint)
	if d, ok := profileCache.Load(key); ok {
		dec := d.(BackendDecision)
		dec.Reason += "（profile 缓存命中）"
		return dec
	}

	f := caps.Forest
	nrows := hint.BatchSize
	if nrows <= 0 {
		nrows = 1
	}
	// 成本上限：rows*cols ≤ 512k cells（保留大批量档的相对形态）。
	if f.NumFeatures > 0 && nrows*f.NumFeatures > 512*1024 {
		nrows = 512 * 1024 / f.NumFeatures
		if nrows < 8 {
			nrows = 8
		}
	}
	vals := syntheticBatch(nrows, f.NumFeatures)
	res := ProfileBackend(caps, vals, nrows, f.NumFeatures, 10)

	// 尊重 hint：无 GPU 意图时把 BornGPU 剔除后重选（次优）。
	pick, rule, reason := res.Pick, res.Rule, res.Reason
	if pick == BackendBornGPU && !hint.HasGPU {
		pick, rule, reason = pickFastest(res.Native, res.BornCPU, BackendTiming{})
	}
	dec := BackendDecision{Backend: pick, Rule: rule,
		Reason: reason + "（LEAVES_BACKEND_PROFILE=1 实测；合成样本 rows=" + strconv.Itoa(nrows) + "）"}

	profileCache.Store(key, dec)
	return dec
}

// syntheticBatch 确定性合成批量（LCG 均匀 [0,1)；仅用于后端间相对计时）。
func syntheticBatch(nrows, ncols int) []float64 {
	vals := make([]float64, nrows*ncols)
	x := uint64(0x9E3779B97F4A7C15)
	for i := range vals {
		x = x*6364136223846793005 + 1442695040888963407
		vals[i] = float64(x>>11) / (1 << 53)
	}
	return vals
}
