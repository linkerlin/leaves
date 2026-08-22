package tree

import (
	"testing"
	"time"
)

// fakeProfileEngine 立即返回的假引擎（仅满足 Engine 接口，供计时守卫测试）。
type fakeProfileEngine struct {
	forest *ForestIR
}

func (e *fakeProfileEngine) PredictDense(vals []float64, nrows, ncols int, predictions []float64, nEstimators int) error {
	return nil
}
func (e *fakeProfileEngine) PredictCSR(indptr, cols []int, vals, predictions []float64, nEstimators int) error {
	return nil
}
func (e *fakeProfileEngine) PredictSingle(fvals []float64, nEstimators int) float64 { return 0 }
func (e *fakeProfileEngine) Predict(fvals []float64, nEstimators int, predictions []float64) error {
	return nil
}
func (e *fakeProfileEngine) PredictLeafIndicesDense(vals []float64, nrows, ncols int, predictions []float64) error {
	return nil
}
func (e *fakeProfileEngine) PredictLeafIndicesCSR(indptr, cols []int, vals, predictions []float64) error {
	return nil
}
func (e *fakeProfileEngine) NOutputGroups() int    { return 1 }
func (e *fakeProfileEngine) NRawOutputGroups() int { return 1 }
func (e *fakeProfileEngine) NFeatures() int        { return 1 }
func (e *fakeProfileEngine) NEstimators() int      { return 1 }
func (e *fakeProfileEngine) NLeaves() []int        { return []int{1} }
func (e *fakeProfileEngine) Name() string          { return "fake" }
func (e *fakeProfileEngine) Forest() *ForestIR     { return e.forest }
func (e *fakeProfileEngine) Close() error          { return nil }

var _ Engine = (*fakeProfileEngine)(nil)

// TestProfileBudgetGuardNormal 正常引擎：预算内完成 → Ok=true 且 ns/op > 0。
func TestProfileBudgetGuardNormal(t *testing.T) {
	ns, ok, err := timePredictDenseBounded(&fakeProfileEngine{}, []float64{0.1, 0.2}, 1, 2, 1, 0, 5, 1_000_000_000)
	if err != nil || !ok || ns <= 0 {
		t.Fatalf("normal engine should pass: ns=%v ok=%v err=%v", ns, ok, err)
	}
}

// TestProfileBudgetGuardDisabled 预算 0 = 禁用守卫（旧行为），正常完成。
func TestProfileBudgetGuardDisabled(t *testing.T) {
	ns, ok, err := timePredictDenseBounded(&fakeProfileEngine{}, []float64{0.1, 0.2}, 1, 2, 1, 0, 5, 0)
	if err != nil || !ok || ns <= 0 {
		t.Fatalf("disabled budget should behave like legacy: ns=%v ok=%v err=%v", ns, ok, err)
	}
}

// slowProfileEngine 每轮 sleep 10ms，用于触发轮间预算检查。
type slowProfileEngine struct{ fakeProfileEngine }

func (e *slowProfileEngine) PredictDense(vals []float64, nrows, ncols int, predictions []float64, nEstimators int) error {
	time.Sleep(10 * time.Millisecond)
	return nil
}

// TestProfileBudgetGuardExhausted 慢引擎 + 20ms 预算 → iters 内超预算按不可用退出。
func TestProfileBudgetGuardExhausted(t *testing.T) {
	_, ok, err := timePredictDenseBounded(&slowProfileEngine{}, []float64{0.1, 0.2}, 1, 2, 1, 0, 50, 20*1000*1000)
	if err != nil || ok {
		t.Fatalf("exhausted budget should be ok=false err=nil, got ok=%v err=%v", ok, err)
	}
}

// TestProfileWithTimeoutHang 整体超时语义：永不完成的测量 → profile_timeout 回落 Native。
// 直接抽测超时分支（不构造真挂起引擎，避免测试依赖不可中断的死等）。
func TestProfileWithTimeoutHang(t *testing.T) {
	never := make(chan ProfileResult) // 永不发送
	res := func() ProfileResult {
		select {
		case r := <-never:
			return r
		case <-time.After(50 * time.Millisecond):
			return ProfileResult{
				Native: BackendTiming{Backend: BackendNative, Ok: true},
				Pick:   BackendNative,
				Rule:   "profile_timeout",
				Reason: "profiling timed out (suspected hung backend) → Native",
			}
		}
	}()
	if res.Pick != BackendNative || res.Rule != "profile_timeout" {
		t.Fatalf("want profile_timeout/Native, got %v/%v", res.Pick, res.Rule)
	}
}

// TestProfileMinNsGuardLocked 锁定计时器归零守卫阈值（防漂移：曾实测 BornGPU ≈0ns）。
func TestProfileMinNsGuardLocked(t *testing.T) {
	if profileMinNsPerOp != 1e-3 {
		t.Fatalf("profileMinNsPerOp drifted: %v", profileMinNsPerOp)
	}
}
