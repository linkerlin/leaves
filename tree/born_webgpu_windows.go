//go:build windows && !js

package tree

import bornwebgpu "github.com/born-ml/born/backend/webgpu"

// BornWebGPUAvailable 当前环境是否可用 Born WebGPU（Windows DX12）。
func BornWebGPUAvailable() bool {
	return bornwebgpu.IsAvailable()
}

func bornOpenWebGPU() (any, error) {
	if !bornwebgpu.IsAvailable() {
		return nil, errBornWebGPUUnavailable
	}
	return bornwebgpu.New()
}

func bornCloseWebGPU(b any) {
	if g, ok := b.(*bornwebgpu.Backend); ok && g != nil {
		g.Release()
	}
}

var errBornWebGPUUnavailable = bornwebgpuErr("webgpu not available")

type bornwebgpuErr string

func (e bornwebgpuErr) Error() string { return "born: " + string(e) }
