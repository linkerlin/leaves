package tree

import (
	"fmt"
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
