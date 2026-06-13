package treebuilder

import (
	"math"
	"testing"
)

func TestRowBinsMatchCPUAccumulation(t *testing.T) {
	dm, idx, grad, hess := synthHistDataset(200, 4)
	gb := BuildGlobalHistBins(dm, 32, nil)
	cuts, numBins, ok := gb.Lookup(0)
	if !ok {
		t.Fatal("no bins for feat 0")
	}
	cfg := Config{
		HistBinPolicy: HistBinGlobal,
		GlobalBins:    gb,
		UseGPUHist:    false,
	}

	cpuG, cpuH := accumulateHist(0, idx, grad, hess, numBins, dm, nil, cuts, cfg)

	rowBins := gb.RowBin(0)
	refG := make([]float64, numBins)
	refH := make([]float64, numBins)
	for _, i := range idx {
		b := rowBins[i]
		refG[b] += grad[i]
		refH[b] += hess[i]
	}
	for i := range refG {
		if math.Abs(refG[i]-cpuG[i]) > 1e-9 || math.Abs(refH[i]-cpuH[i]) > 1e-9 {
			t.Fatalf("bin %d: refG=%v cpuG=%v refH=%v cpuH=%v", i, refG[i], cpuG[i], refH[i], cpuH[i])
		}
	}
}

func TestBatchGPUHistMultipleFeats(t *testing.T) {
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

	feats := []int{0, 1, 2, 3}
	sumG, sumH := sumGradHess(idx, grad, hess)
	batch := batchAccumulateHistWebGPU(feats, idx, grad, hess, sumG, sumH, cfg.Lambda, cfg)
	if batch == nil || len(batch) == 0 {
		t.Skip("webgpu batch unavailable")
	}
	for _, f := range feats {
		gr, ok := batch[f]
		if !ok || !gr.ok {
			continue
		}
		cuts, numBins, ok2 := gb.Lookup(f)
		if !ok2 {
			continue
		}
		cpuG, cpuH := accumulateHistCPU(f, idx, grad, hess, numBins, dm, nil, cuts, cpuCfg)
		if gr.hasGain {
			sumG, sumH := sumGradHess(idx, grad, hess)
			sCPU, gCPU := scanHistGainsCPU(cpuG, cpuH, sumG, sumH, cfg.Lambda)
			if gr.splitIdx != sCPU || math.Abs(gr.gain-gCPU) > 1e-3 {
				t.Fatalf("feat %d fused gain: gpu=(%d,%v) cpu=(%d,%v)", f, gr.splitIdx, gr.gain, sCPU, gCPU)
			}
			continue
		}
		for i := range cpuG {
			if math.Abs(cpuG[i]-gr.histG[i]) > 1e-3 || math.Abs(cpuH[i]-gr.histH[i]) > 1e-3 {
				t.Fatalf("feat %d bin %d: cpuG=%v gpuG=%v", f, i, cpuG[i], gr.histG[i])
			}
		}
	}
}

func TestFindBestHistSplitGPUBatch(t *testing.T) {
	if !WebGPUHistAvailable() {
		t.Skip("webgpu unavailable")
	}
	dm, idx, grad, hess := synthHistDataset(200, 8)
	cfg := Config{
		MaxDepth:      4,
		LearningRate:  0.3,
		Lambda:        1.0,
		MaxBin:        64,
		Gamma:         0,
		HistBinPolicy: HistBinGlobal,
		GlobalBins:    BuildGlobalHistBins(dm, 64, nil),
		UseGPUHist:    true,
		AccelMode:     AccelModeWebGPU,
		NumThreads:    4,
	}
	row := make([]float64, dm.NumCol())
	pick := findBestHistSplit(dm, idx, []int{0, 1, 2, 3, 4, 5, 6, 7}, grad, hess, 0, 0, row, 0, cfg)
	_ = pick
	treeIR := BuildHist(dm, idx, grad, hess, cfg)
	if treeIR == nil {
		t.Fatal("nil tree")
	}
}

func TestGPUHistBuildMatchesCPU(t *testing.T) {
	if !WebGPUHistAvailable() {
		t.Skip("webgpu unavailable")
	}
	dm, idx, grad, hess := synthHistDataset(300, 6)
	gb := BuildGlobalHistBins(dm, 64, nil)
	cuts, numBins, ok := gb.Lookup(1)
	if !ok || numBins < bornHistMinBins {
		t.Skip("feature bins too small for gpu hist test")
	}
	cfg := Config{
		HistBinPolicy: HistBinGlobal,
		GlobalBins:    gb,
		UseGPUHist:    true,
		NumThreads:    1,
		AccelMode:     AccelModeWebGPU,
	}
	cpuCfg := cfg
	cpuCfg.UseGPUHist = false

	cpuG, cpuH := accumulateHist(1, idx, grad, hess, numBins, dm, nil, cuts, cpuCfg)
	sumG, sumH := sumGradHess(idx, grad, hess)
	batch := batchAccumulateHistWebGPU([]int{1}, idx, grad, hess, sumG, sumH, cfg.Lambda, cfg)
	if batch == nil {
		t.Skip("webgpu hist build unavailable")
	}
	gr, found := batch[1]
	if !found || !gr.ok {
		t.Skip("webgpu hist build unavailable")
	}
	if gr.hasGain {
		sCPU, gCPU := scanHistGainsCPU(cpuG, cpuH, sumG, sumH, cfg.Lambda)
		if gr.splitIdx != sCPU || math.Abs(gr.gain-gCPU) > 1e-3 {
			t.Fatalf("fused gain: gpu=(%d,%v) cpu=(%d,%v)", gr.splitIdx, gr.gain, sCPU, gCPU)
		}
		return
	}
	gpuG, gpuH := gr.histG, gr.histH
	gpuOK := true
	_ = gpuOK
	for i := range cpuG {
		if math.Abs(cpuG[i]-gpuG[i]) > 1e-3 || math.Abs(cpuH[i]-gpuH[i]) > 1e-3 {
			t.Fatalf("bin %d: cpuG=%v gpuG=%v cpuH=%v gpuH=%v", i, cpuG[i], gpuG[i], cpuH[i], gpuH[i])
		}
	}
}
