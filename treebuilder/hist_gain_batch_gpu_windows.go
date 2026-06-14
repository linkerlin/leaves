//go:build windows && !js

package treebuilder

import (
	bornwebgpu "github.com/born-ml/born/backend/webgpu"
	"github.com/born-ml/born/tensor"
)

// batchGainScanHistF32OnGPU 将多行直方图一次上传，2D 向量化算增益后逐行 argmax。
func batchGainScanHistF32OnGPU(
	gpu *bornwebgpu.Backend,
	rows []gpuHistF32Row,
	sumG, sumH, lambda float64,
) map[int]gpuGainPick {
	out := make(map[int]gpuGainPick, len(rows))
	if len(rows) == 0 {
		return out
	}
	if len(rows) == 1 {
		row := rows[0]
		prefix := row.numBins - 1
		if prefix < 1 {
			return out
		}
		split, gain, ok := gainScanHistF32OnGPU(gpu, row.histG[:prefix], row.histH[:prefix], sumG, sumH, lambda)
		if ok {
			out[row.feat] = gpuGainPick{splitIdx: split, gain: gain, ok: true}
		}
		return out
	}

	maxPrefix := 0
	for _, row := range rows {
		if p := row.numBins - 1; p > maxPrefix {
			maxPrefix = p
		}
	}
	if maxPrefix < 1 {
		return out
	}

	F := len(rows)
	gLeft := make([]float32, F*maxPrefix)
	hLeft := make([]float32, F*maxPrefix)
	validPrefix := make([]int, F)
	for f, row := range rows {
		prefix := row.numBins - 1
		validPrefix[f] = prefix
		if prefix < 1 {
			continue
		}
		var gAcc, hAcc float32
		base := f * maxPrefix
		for s := 0; s < prefix; s++ {
			gAcc += row.histG[s]
			hAcc += row.histH[s]
			gLeft[base+s] = gAcc
			hLeft[base+s] = hAcc
		}
	}

	gLeftT, err := tensor.FromSlice(gLeft, tensor.Shape{F, maxPrefix}, gpu)
	if err != nil {
		return out
	}
	hLeftT, err := tensor.FromSlice(hLeft, tensor.Shape{F, maxPrefix}, gpu)
	if err != nil {
		return out
	}

	sumG32 := float32(sumG)
	sumH32 := float32(sumH)
	lambda32 := float32(lambda)
	sumGT := tensor.Full[float32](tensor.Shape{F, maxPrefix}, sumG32, gpu)
	sumHT := tensor.Full[float32](tensor.Shape{F, maxPrefix}, sumH32, gpu)
	lambdaT := tensor.Full[float32](tensor.Shape{F, maxPrefix}, lambda32, gpu)
	gRightT := sumGT.Sub(gLeftT)
	hRightT := sumHT.Sub(hLeftT)
	hLeftL := hLeftT.Add(lambdaT)
	hRightL := hRightT.Add(lambdaT)
	leftTerm := gLeftT.Mul(gLeftT).Div(hLeftL)
	rightTerm := gRightT.Mul(gRightT).Div(hRightL)
	total := float32((sumG * sumG) / (sumH + lambda))
	totalTerm := tensor.Full[float32](tensor.Shape{F, maxPrefix}, total, gpu)
	gainsT := leftTerm.Add(rightTerm).Sub(totalTerm).MulScalar(0.5)

	gainData := gainsT.Data()
	hLData := hLeftT.Data()
	hRData := hRightT.Data()

	for f, row := range rows {
		prefix := validPrefix[f]
		if prefix < 1 {
			continue
		}
		base := f * maxPrefix
		rowGain := f32ToF64(gainData[base : base+maxPrefix])
		rowHL := f32ToF64(hLData[base : base+maxPrefix])
		rowHR := f32ToF64(hRData[base : base+maxPrefix])
		for s := prefix; s < maxPrefix; s++ {
			rowHL[s] = 0
			rowHR[s] = 0
		}
		split, gain, err := bestSplitFromGainData(rowGain, rowHL, rowHR)
		if err != nil || split < 0 {
			continue
		}
		if split >= prefix {
			continue
		}
		out[row.feat] = gpuGainPick{splitIdx: split, gain: gain, ok: true}
	}
	return out
}
