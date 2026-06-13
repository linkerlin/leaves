package treebuilder

import "testing"

func TestResolveEffectiveAccelMode(t *testing.T) {
	cases := []struct {
		req      string
		nRow     int
		webgpu   bool
		want     string
	}{
		{AccelModeAuto, 1000, true, AccelModeCPU},
		{AccelModeAuto, 30000, true, AccelModeWebGPU},
		{AccelModeAuto, 50000, false, AccelModeCPU},
		{AccelModeWebGPU, 100, true, AccelModeWebGPU},
		{AccelModeCPU, 100000, true, AccelModeCPU},
		{"", 50000, true, AccelModeWebGPU},
	}
	for _, c := range cases {
		got := ResolveEffectiveAccelMode(c.req, c.nRow, c.webgpu)
		if got != c.want {
			t.Errorf("req=%q nRow=%d webgpu=%v: got %q want %q", c.req, c.nRow, c.webgpu, got, c.want)
		}
	}
}
