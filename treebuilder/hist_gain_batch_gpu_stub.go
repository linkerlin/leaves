//go:build !windows

package treebuilder

func batchGainScanHistF32OnGPU(
	gpu interface{},
	rows []gpuHistF32Row,
	sumG, sumH, lambda float64,
) map[int]gpuGainPick {
	_ = gpu
	_ = rows
	_ = sumG
	_ = sumH
	_ = lambda
	return nil
}
