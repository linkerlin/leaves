package treebuilder

import (
	"math"
	"testing"
)

func TestBatchGainScanMatchesCPU(t *testing.T) {
	if !WebGPUHistAvailable() {
		t.Skip("webgpu unavailable")
	}
	dm, idx, grad, hess := synthHistDataset(300, 8)
	gb := BuildGlobalHistBins(dm, 64, nil)
	cfg := Config{
		HistBinPolicy: HistBinGlobal,
		GlobalBins:    gb,
		UseGPUHist:    true,
		AccelMode:     AccelModeWebGPU,
	}
	cpuCfg := cfg
	cpuCfg.UseGPUHist = false

	feats := []int{0, 1, 2, 3, 4, 5}
	sumG, sumH := sumGradHess(idx, grad, hess)
	batch := batchAccumulateHistWebGPU(feats, idx, grad, hess, sumG, sumH, cfg.Lambda, cfg)
	if batch == nil || len(batch) == 0 {
		t.Skip("webgpu batch unavailable")
	}
	for _, f := range feats {
		gr, ok := batch[f]
		if !ok || !gr.ok || !gr.hasGain {
			t.Fatalf("feat %d: expected batched gain, got ok=%v hasGain=%v", f, gr.ok, gr.hasGain)
		}
		cuts, numBins, ok2 := gb.Lookup(f)
		if !ok2 {
			continue
		}
		cpuG, cpuH := accumulateHistCPU(f, idx, grad, hess, numBins, dm, nil, cuts, cpuCfg)
		sCPU, gCPU := scanHistGainsCPU(cpuG, cpuH, sumG, sumH, cfg.Lambda)
		if gr.splitIdx != sCPU || math.Abs(gr.gain-gCPU) > 1e-3 {
			t.Fatalf("feat %d batch gain: gpu=(%d,%v) cpu=(%d,%v)", f, gr.splitIdx, gr.gain, sCPU, gCPU)
		}
	}
}
