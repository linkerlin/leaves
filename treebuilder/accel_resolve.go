package treebuilder

// AccelWebGPUMinRows auto 模式下启用 WebGPU hist/gain 批量的最小训练行数（本机 5 万行 benchmark 交叉点 ~3 万）。
const AccelWebGPUMinRows = 30000

// ResolveEffectiveAccelMode 解析实际加速模式；auto 时按行数与 WebGPU 可用性选择 cpu 或 webgpu。
func ResolveEffectiveAccelMode(requested string, nRow int, webgpuAvail bool) string {
	if requested == "" {
		requested = AccelModeFromEnv()
	}
	switch requested {
	case AccelModeWebGPU, AccelModeBornCPU, AccelModeCPU:
		return requested
	case AccelModeAuto:
		if webgpuAvail && nRow >= AccelWebGPUMinRows {
			return AccelModeWebGPU
		}
		return AccelModeCPU
	default:
		return AccelModeAuto
	}
}
