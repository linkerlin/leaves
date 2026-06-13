package treebuilder

import (
	"sync"

	borncpu "github.com/born-ml/born/backend/cpu"
	"github.com/born-ml/born/tensor"

	"github.com/dmitryikh/leaves/data"
	"github.com/dmitryikh/leaves/tree"
)

const bornHistMinBins = 16

var (
	bornCPUOnce sync.Once
	bornCPU     *borncpu.Backend
)

func initBornCPU() {
	bornCPUOnce.Do(func() {
		bornCPU = borncpu.New()
	})
}

// BornHistAvailable Born CPU hist 增益扫描是否可用（born 已在 go.mod 依赖中）。
func BornHistAvailable() bool { return true }

// WebGPUHistAvailable 当前环境是否可尝试 WebGPU hist 增益扫描。
func WebGPUHistAvailable() bool {
	return tree.BornWebGPUAvailable()
}

func scanHistGains(histG, histH []float64, sumG, sumH, lambda float64, cfg Config) (int, float64) {
	n := len(histG)
	mode := effectiveAccelMode(cfg)
	tryWebGPU := cfg.UseGPUHist && mode == AccelModeWebGPU && cfg.NumThreads == 1
	tryBorn := tryBornGainScan(cfg)
	if mode == AccelModeCPU {
		tryWebGPU, tryBorn = false, false
	}

	if tryWebGPU && n >= bornHistMinBins {
		if split, gain, ok := scanHistGainsWebGPU(histG, histH, sumG, sumH, lambda); ok {
			setAccelWebGPUOK(true)
			recordGainScanWebGPU()
			return split, gain
		}
	}
	if tryBorn && n >= bornHistMinBins && BornHistAvailable() {
		initBornCPU()
		if split, gain, err := scanHistGainsBorn(bornCPU, histG, histH, sumG, sumH, lambda); err == nil {
			recordGainScanBornCPU()
			return split, gain
		}
	}
	recordGainScanPureCPU()
	return scanHistGainsCPU(histG, histH, sumG, sumH, lambda)
}

func tryBornGainScan(cfg Config) bool {
	mode := effectiveAccelMode(cfg)
	switch mode {
	case AccelModeCPU:
		return false
	case AccelModeBornCPU:
		return true
	case AccelModeAuto:
		return cfg.UseGPUHist
	default:
		return false
	}
}

// scanHistGainsBorn Born CPU 向量化增益扫描（与纯 CPU 算法等价）。
func scanHistGainsBorn(b *borncpu.Backend, histG, histH []float64, sumG, sumH, lambda float64) (int, float64, error) {
	n := len(histG)
	if n < 2 {
		return -1, 0, nil
	}
	prefix := n - 1
	gPrefix, err := tensor.FromSlice(histG[:prefix], tensor.Shape{prefix}, b)
	if err != nil {
		return -1, 0, err
	}
	hPrefix, err := tensor.FromSlice(histH[:prefix], tensor.Shape{prefix}, b)
	if err != nil {
		return -1, 0, err
	}
	gLeft := cumsum1D(b, gPrefix.Data())
	hLeft := cumsum1D(b, hPrefix.Data())
	sumGT := tensor.Full[float64](tensor.Shape{prefix}, sumG, b)
	sumHT := tensor.Full[float64](tensor.Shape{prefix}, sumH, b)
	lambdaT := tensor.Full[float64](tensor.Shape{prefix}, lambda, b)
	gRight := sumGT.Sub(gLeft)
	hRight := sumHT.Sub(hLeft)
	hLeftL := hLeft.Add(lambdaT)
	hRightL := hRight.Add(lambdaT)
	leftTerm := gLeft.Mul(gLeft).Div(hLeftL)
	rightTerm := gRight.Mul(gRight).Div(hRightL)
	totalTerm := tensor.Full[float64](tensor.Shape{prefix}, (sumG*sumG)/(sumH+lambda), b)
	gains := leftTerm.Add(rightTerm).Sub(totalTerm).MulScalar(0.5)
	return bestSplitFromGainData(gains.Data(), hLeft.Data(), hRight.Data())
}

func cumsum1D(b *borncpu.Backend, data []float64) *tensor.Tensor[float64, *borncpu.Backend] {
	out := make([]float64, len(data))
	var acc float64
	for i, v := range data {
		acc += v
		out[i] = acc
	}
	res, err := tensor.FromSlice(out, tensor.Shape{len(out)}, b)
	if err != nil {
		panic(err)
	}
	return res
}

func bestSplitFromGainData(gainData, hLeft, hRight []float64) (int, float64, error) {
	prefix := len(gainData)
	bestSplit := -1
	bestGain := 0.0
	for s := 0; s < prefix; s++ {
		if hLeft[s] <= 0 || hRight[s] <= 0 {
			continue
		}
		g := gainData[s]
		if g > bestGain {
			bestGain = g
			bestSplit = s
		}
	}
	return bestSplit, bestGain, nil
}

// BuildHistGPU gpu_hist：启用 WebGPU hist 增益扫描（不可用则自动回退 Born CPU / 纯 CPU）。
func BuildHistGPU(dm data.Matrix, indices []int, grad, hess []float64, cfg Config) *tree.TreeIR {
	cfg.UseGPUHist = true
	return BuildHist(dm, indices, grad, hess, cfg)
}
