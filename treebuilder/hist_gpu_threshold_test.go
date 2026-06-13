package treebuilder

import "testing"

func TestGPUHistMinSamplesForDepth(t *testing.T) {
	cases := []struct {
		depth int
		want  int
	}{
		{0, 64},
		{4, 64},
		{5, 48},
		{6, 32},
		{10, 32},
	}
	for _, c := range cases {
		if got := gpuHistMinSamplesForDepth(c.depth); got != c.want {
			t.Errorf("depth %d: got %d want %d", c.depth, got, c.want)
		}
	}
}

func TestFilterGPUHistFeatsDepth(t *testing.T) {
	dm, idx, grad, hess := synthHistDataset(40, 4)
	gb := BuildGlobalHistBins(dm, 32, nil)
	cfg := Config{
		HistBinPolicy: HistBinGlobal,
		GlobalBins:    gb,
		UseGPUHist:    true,
		AccelMode:     AccelModeWebGPU,
	}
	feats := []int{0, 1, 2, 3}
	if got := filterGPUHistFeats(feats, idx, 0, cfg); len(got) != 0 {
		t.Fatalf("depth 0 needs 64 samples, got %d gpu feats", len(got))
	}
	if got := filterGPUHistFeats(feats, idx, 4, cfg); len(got) != 0 {
		t.Fatalf("depth 4 needs 64 samples, got %d gpu feats", len(got))
	}
	if got := filterGPUHistFeats(feats, idx, 6, cfg); len(got) != len(feats) {
		t.Fatalf("depth 6 with 40 samples: got %d gpu feats want %d", len(got), len(feats))
	}
	_ = grad
	_ = hess
}
