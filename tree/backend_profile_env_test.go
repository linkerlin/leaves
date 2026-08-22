package tree

import (
	"strings"
	"testing"
)

// TestBackendProfileEnvOptIn LEAVES_BACKEND_PROFILE=1 时 Auto 选型走实测路径
// （Rule=profile_*，Reason 带实测数字），且同形状类第二次命中缓存。
func TestBackendProfileEnvOptIn(t *testing.T) {
	t.Setenv("LEAVES_BACKEND_PROFILE", "1")
	f := makeForest()
	caps := ModelCapsFromForest(f, false, true)
	hint := WorkloadHint{BatchSize: 128}

	d := SelectBackendExplained(caps, hint)
	if !strings.HasPrefix(d.Rule, "profile_") {
		t.Fatalf("rule: got %q want profile_*", d.Rule)
	}
	if !strings.Contains(d.Reason, "ns/op") {
		t.Fatalf("reason missing measured numbers: %q", d.Reason)
	}
	if d.Backend != BackendNative && d.Backend != BackendBornCPU && d.Backend != BackendBornGPU {
		t.Fatalf("backend pick unexpected: %v", d.Backend)
	}

	d2 := SelectBackendExplained(caps, hint)
	if !strings.Contains(d2.Reason, "缓存命中") {
		t.Fatalf("second call should hit cache: %q", d2.Reason)
	}
	if d2.Backend != d.Backend || d2.Rule != d.Rule {
		t.Fatalf("cache drift: %v/%v vs %v/%v", d2.Backend, d2.Rule, d.Backend, d.Rule)
	}
}

// TestBackendProfileEnvOff 默认（未设 env）2.1 决策表行为：任意 batch 均 Native。
func TestBackendProfileEnvOff(t *testing.T) {
	f := makeForest()
	caps := ModelCapsFromForest(f, false, true)
	d := SelectBackendExplained(caps, WorkloadHint{BatchSize: 1})
	if d.Rule != "small_batch" {
		t.Fatalf("rule: got %q want small_batch", d.Rule)
	}
	d = SelectBackendExplained(caps, WorkloadHint{BatchSize: 128})
	if d.Rule != "native_batch" {
		t.Fatalf("rule: got %q want native_batch", d.Rule)
	}
}

// TestBackendProfileEnvOffTruthy 只认 1/on/true/yes；垃圾值不触发。
func TestBackendProfileEnvOffTruthy(t *testing.T) {
	t.Setenv("LEAVES_BACKEND_PROFILE", "maybe")
	f := makeForest()
	caps := ModelCapsFromForest(f, false, true)
	d := SelectBackendExplained(caps, WorkloadHint{BatchSize: 128})
	if d.Rule != "native_batch" {
		t.Fatalf("rule: got %q want native_batch (env garbage must not trigger)", d.Rule)
	}
}
