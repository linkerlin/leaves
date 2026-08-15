//go:build windows && !js

package tree

import (
	"os"
	"strings"

	bornwebgpu "github.com/born-ml/born/backend/webgpu"
)

// BornWebGPUAvailable 当前环境是否可用 Born WebGPU（Windows DX12）。
// LEAVES_BORN_GPU=0|off|false 可强制关闭（如 CI 的 WARP 设备会运行时 DEVICE_REMOVED）。
func BornWebGPUAvailable() bool {
	switch strings.ToLower(strings.TrimSpace(os.Getenv("LEAVES_BORN_GPU"))) {
	case "0", "off", "false":
		return false
	}
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
