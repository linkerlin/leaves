//go:build windows

package treebuilder

import (
	"log"
	"sync"

	bornwebgpu "github.com/born-ml/born/backend/webgpu"
	"github.com/born-ml/born/tensor"
)

var (
	trainWebGPUOnce     sync.Once
	trainWebGPU         *bornwebgpu.Backend
	trainWebGPUOK       bool
	trainWebGPULog      bool
	trainWebGPUMu       sync.Mutex
	trainWebGPUDisabled bool
)

func initTrainWebGPU() (*bornwebgpu.Backend, bool) {
	trainWebGPUOnce.Do(func() {
		if !WebGPUHistAvailable() {
			if !trainWebGPULog {
				trainWebGPULog = true
				log.Printf("[leaves/train] accel: webgpu unavailable on this platform, fallback born_cpu/pure_cpu")
			}
			return
		}
		gpu, err := bornwebgpu.New()
		if err != nil {
			if !trainWebGPULog {
				trainWebGPULog = true
				log.Printf("[leaves/train] accel: webgpu init failed (%v), fallback born_cpu/pure_cpu", err)
			}
			return
		}
		trainWebGPU = gpu
		trainWebGPUOK = true
		if !trainWebGPULog {
			trainWebGPULog = true
			log.Printf("[leaves/train] accel: webgpu hist batch + batched gain scan enabled")
		}
	})
	return trainWebGPU, trainWebGPUOK
}

func scanHistGainsWebGPU(histG, histH []float64, sumG, sumH, lambda float64) (split int, gain float64, ok bool) {
	if trainWebGPUDisabled {
		return -1, 0, false
	}
	defer func() {
		if r := recover(); r != nil {
			trainWebGPUDisabled = true
			log.Printf("[leaves/train] accel: webgpu scan error (%v), disabled for session; fallback born_cpu/pure_cpu", r)
			split, gain, ok = -1, 0, false
		}
	}()

	trainWebGPUMu.Lock()
	defer trainWebGPUMu.Unlock()

	gpu, ready := initTrainWebGPU()
	if !ready || gpu == nil {
		return -1, 0, false
	}
	n := len(histG)
	if n < 2 {
		return -1, 0, false
	}
	prefix := n - 1
	return gainScanHistF32OnGPU(gpu, f64ToF32(histG[:prefix]), f64ToF32(histH[:prefix]), sumG, sumH, lambda)
}

func gainScanHistF32OnGPU(
	gpu *bornwebgpu.Backend,
	histG, histH []float32,
	sumG, sumH, lambda float64,
) (split int, gain float64, ok bool) {
	prefix := len(histG)
	if prefix < 1 {
		return -1, 0, false
	}

	gPrefix, err := tensor.FromSlice(histG, tensor.Shape{prefix}, gpu)
	if err != nil {
		return -1, 0, false
	}
	hPrefix, err := tensor.FromSlice(histH, tensor.Shape{prefix}, gpu)
	if err != nil {
		return -1, 0, false
	}
	gLeft := cumsum1DF32(gpu, gPrefix.Data())
	hLeft := cumsum1DF32(gpu, hPrefix.Data())
	sumG32 := float32(sumG)
	sumH32 := float32(sumH)
	lambda32 := float32(lambda)
	sumGT := tensor.Full[float32](tensor.Shape{prefix}, sumG32, gpu)
	sumHT := tensor.Full[float32](tensor.Shape{prefix}, sumH32, gpu)
	lambdaT := tensor.Full[float32](tensor.Shape{prefix}, lambda32, gpu)
	gRight := sumGT.Sub(gLeft)
	hRight := sumHT.Sub(hLeft)
	hLeftL := hLeft.Add(lambdaT)
	hRightL := hRight.Add(lambdaT)
	leftTerm := gLeft.Mul(gLeft).Div(hLeftL)
	rightTerm := gRight.Mul(gRight).Div(hRightL)
	total := float32((sumG * sumG) / (sumH + lambda))
	totalTerm := tensor.Full[float32](tensor.Shape{prefix}, total, gpu)
	gains := leftTerm.Add(rightTerm).Sub(totalTerm).MulScalar(0.5)

	hL := f32ToF64(hLeft.Data())
	hR := f32ToF64(hRight.Data())
	split, gain, err = bestSplitFromGainData(f32ToF64(gains.Data()), hL, hR)
	if err != nil {
		return -1, 0, false
	}
	return split, gain, true
}

func cumsum1DF32(b *bornwebgpu.Backend, data []float32) *tensor.Tensor[float32, *bornwebgpu.Backend] {
	out := make([]float32, len(data))
	var acc float32
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

func f64ToF32(in []float64) []float32 {
	out := make([]float32, len(in))
	for i, v := range in {
		out[i] = float32(v)
	}
	return out
}

func f32ToF64(in []float32) []float64 {
	out := make([]float64, len(in))
	for i, v := range in {
		out[i] = float64(v)
	}
	return out
}
