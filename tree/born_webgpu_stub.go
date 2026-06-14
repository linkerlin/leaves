//go:build !windows && !js

package tree

import "fmt"

var errBornWebGPUUnavailable = fmt.Errorf("born: webgpu requires windows")

// BornWebGPUAvailable 非 Windows 平台无 WebGPU 推理。
func BornWebGPUAvailable() bool {
	return false
}

func bornOpenWebGPU() (any, error) {
	return nil, errBornWebGPUUnavailable
}

func bornCloseWebGPU(any) {}
