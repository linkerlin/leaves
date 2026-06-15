package train

import "github.com/linkerlin/leaves/treebuilder"

// ResolveTrainTreeMethodWithAccel 解析建树算法与 GPU hist 开关；auto 加速在大数据 + WebGPU 时升级为 gpu_hist。
func ResolveTrainTreeMethodWithAccel(
	requestedMethod string,
	nRow int,
	effectiveAccel string,
	webgpuAvail bool,
) (resolved string, useGPUHist bool) {
	if requestedMethod == "" {
		requestedMethod = treebuilder.MethodAuto
	}
	resolved, useGPUHist = ResolveTrainTreeMethod(requestedMethod, nRow)
	if effectiveAccel != treebuilder.AccelModeWebGPU || !webgpuAvail || nRow < treebuilder.AccelWebGPUMinRows {
		return resolved, useGPUHist
	}
	if resolved == treebuilder.MethodExact {
		return resolved, false
	}
	switch requestedMethod {
	case treebuilder.MethodAuto, "", treebuilder.MethodGPUHist:
		return treebuilder.MethodGPUHist, true
	case treebuilder.MethodHist:
		// 显式 hist：保持 CPU hist，不强行升级
		return treebuilder.MethodHist, false
	default:
		return resolved, useGPUHist
	}
}
