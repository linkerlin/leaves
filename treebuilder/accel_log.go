package treebuilder

import (
	"fmt"
	"log"
	"sync"
)

// AccelStats 训练 hist 加速路径计数。
type AccelStats struct {
	GainScanWebGPU  int
	GainScanBornCPU int
	GainScanPureCPU int
	HistBuildCPU    int
	HistBuildWebGPU int
	WebGPUOK        bool
}

var (
	accelMu     sync.Mutex
	accelStats  AccelStats
	accelInited bool
)

// ResetAccelStats 新一轮训练前清零统计。
func ResetAccelStats() {
	accelMu.Lock()
	accelStats = AccelStats{}
	accelInited = false
	accelMu.Unlock()
}

func recordGainScanWebGPU() {
	accelMu.Lock()
	accelStats.GainScanWebGPU++
	accelMu.Unlock()
}

func recordGainScanWebGPUBatch(n int) {
	if n <= 0 {
		return
	}
	accelMu.Lock()
	accelStats.GainScanWebGPU += n
	accelMu.Unlock()
}

func recordGainScanBornCPU() {
	accelMu.Lock()
	accelStats.GainScanBornCPU++
	accelMu.Unlock()
}

func recordGainScanPureCPU() {
	accelMu.Lock()
	accelStats.GainScanPureCPU++
	accelMu.Unlock()
}

func recordHistBuildCPU() {
	accelMu.Lock()
	accelStats.HistBuildCPU++
	accelMu.Unlock()
}

func recordHistBuildWebGPU() {
	accelMu.Lock()
	accelStats.HistBuildWebGPU++
	accelMu.Unlock()
}

func setAccelWebGPUOK(ok bool) {
	accelMu.Lock()
	accelStats.WebGPUOK = ok
	accelMu.Unlock()
}

// SnapshotAccelStats 返回当前加速路径计数副本。
func SnapshotAccelStats() AccelStats {
	accelMu.Lock()
	s := accelStats
	accelMu.Unlock()
	return s
}

// AccelSummary 返回本轮训练加速路径摘要。
func AccelSummary() string {
	s := SnapshotAccelStats()
	scanTotal := s.GainScanWebGPU + s.GainScanBornCPU + s.GainScanPureCPU
	histTotal := s.HistBuildCPU + s.HistBuildWebGPU
	gpuNote := "inactive"
	if s.WebGPUOK {
		gpuNote = "active"
	}
	return fmt.Sprintf(
		"webgpu=%s gain_scan(webgpu=%d born_cpu=%d pure_cpu=%d total=%d) hist_build(cpu=%d webgpu=%d total=%d)",
		gpuNote,
		s.GainScanWebGPU, s.GainScanBornCPU, s.GainScanPureCPU, scanTotal,
		s.HistBuildCPU, s.HistBuildWebGPU, histTotal,
	)
}

// LogTrainAccelStart 训练开始时输出加速解析结果（每轮 Fit 一次）。
func LogTrainAccelStart(requested, resolved, accelRequested, accelEffective string, useGPU bool, nRow int, webgpuAvail, bornAvail bool) {
	accelMu.Lock()
	if !accelInited {
		accelInited = true
		log.Printf(
			"[leaves/train] accel: requested=%q resolved=%q accel_requested=%q accel_effective=%q use_gpu_hist=%v rows=%d webgpu_available=%v born_hist=%v",
			requested, resolved, accelRequested, accelEffective, useGPU, nRow, webgpuAvail, bornAvail,
		)
	}
	accelMu.Unlock()
}

// LogTrainAccelEnd 训练结束输出扫描统计。
func LogTrainAccelEnd() {
	log.Printf("[leaves/train] accel summary: %s", AccelSummary())
}
