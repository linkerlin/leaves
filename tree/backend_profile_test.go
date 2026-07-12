package tree

import "testing"

// TestProfileBackendPicks ProfileBackend 对小森林计时并给出可解释推荐。
// 不断言具体 Pick（机器相关），只断言可解释性与一致性。
func TestProfileBackendPicks(t *testing.T) {
	f := makeForest()
	caps := ModelCapsFromForest(f, false, true) // 数值树
	vals := []float64{
		0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6,
		0.15, 0.85, 0.25, 0.75, 0.35, 0.65, 0.45, 0.55,
	} // 8 行 × 2 列
	res := ProfileBackend(caps, vals, 8, 2, 10)

	if !res.Native.Ok {
		t.Fatal("Native timing should always be Ok")
	}
	// Pick 必为某个 Ok 后端，且与其 NsPerOp 一致（最快）。
	okByName := map[Backend]BackendTiming{}
	if res.Native.Ok {
		okByName[res.Native.Backend] = res.Native
	}
	if res.BornCPU.Ok {
		okByName[res.BornCPU.Backend] = res.BornCPU
	}
	if res.BornGPU.Ok {
		okByName[res.BornGPU.Backend] = res.BornGPU
	}
	pickT, ok := okByName[res.Pick]
	if !ok {
		t.Fatalf("Pick=%v 不是已 Ok 的后端（Ok 集 = %v）", res.Pick, okByName)
	}
	for _, tt := range okByName {
		if tt.NsPerOp < pickT.NsPerOp-1e-9 {
			t.Fatalf("Pick=%v ns=%g 但 %v 更快 ns=%g", res.Pick, pickT.NsPerOp, tt.Backend, tt.NsPerOp)
		}
	}
	// 可解释：Rule 码 + Reason 含 ns/op。
	if res.Rule == "" {
		t.Fatal("empty Rule")
	}
	switch res.Rule {
	case "profile_native", "profile_born_cpu", "profile_born_gpu":
	default:
		t.Fatalf("unexpected Rule=%q", res.Rule)
	}
	if res.Reason == "" {
		t.Fatal("empty Reason")
	}
}

// TestProfileBackendInvalid 非法输入回落 Native + profile_invalid。
func TestProfileBackendInvalid(t *testing.T) {
	res := ProfileBackend(ModelCaps{}, nil, 0, 0, 5)
	if res.Pick != BackendNative || res.Rule != "profile_invalid" {
		t.Fatalf("expected Native/profile_invalid, got Pick=%v Rule=%q", res.Pick, res.Rule)
	}
}

// TestProfileBackendNoBornOnUnsupported 含 cat-small 时 Born 不支持 → 仅 Native Ok。
func TestProfileBackendNoBornOnUnsupported(t *testing.T) {
	f := makeForest()
	// 注入 cat-small → BornSupports=false
	f.Trees[0].CatSmall = []bool{true}
	caps := ModelCapsFromForest(f, false, true)
	vals := []float64{0.1, 0.9, 0.2, 0.8}
	res := ProfileBackend(caps, vals, 2, 2, 5)
	if !res.Native.Ok {
		t.Fatal("Native should be Ok")
	}
	if res.BornCPU.Ok {
		t.Fatal("BornCPU should NOT be Ok on cat-small forest")
	}
	if res.Pick != BackendNative {
		t.Fatalf("Pick=%v want Native", res.Pick)
	}
}
