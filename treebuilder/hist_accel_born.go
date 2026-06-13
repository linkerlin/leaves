//go:build born_train

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

func BornHistAvailable() bool { return true }

func scanHistGains(histG, histH []float64, sumG, sumH, lambda float64, useGPU bool) (int, float64) {
	_ = useGPU
	n := len(histG)
	if n < bornHistMinBins {
		return scanHistGainsCPU(histG, histH, sumG, sumH, lambda)
	}
	initBornCPU()
	bestS, bestG, err := scanHistGainsBorn(bornCPU, histG, histH, sumG, sumH, lambda)
	if err != nil {
		return scanHistGainsCPU(histG, histH, sumG, sumH, lambda)
	}
	return bestS, bestG
}

// scanHistGainsBorn 用 Born 张量在 CPU 上向量化增益扫描（与 CPU 算法等价）。
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
	gainData := gains.Data()
	bestSplit := -1
	bestGain := 0.0
	for s := 0; s < prefix; s++ {
		hL := hLeft.Data()[s]
		hR := hRight.Data()[s]
		if hL <= 0 || hR <= 0 {
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

func BuildHistGPU(dm data.Matrix, indices []int, grad, hess []float64, cfg Config) *tree.TreeIR {
	_ = dm
	_ = indices
	_ = grad
	_ = hess
	_ = cfg
	return nil
}
