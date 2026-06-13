package treebuilder

import (
	"os"
	"strings"
)

const (
	AccelModeAuto    = "auto"
	AccelModeWebGPU  = "webgpu"
	AccelModeBornCPU = "born_cpu"
	AccelModeCPU     = "cpu"
)

// AccelModeFromEnv 读取 LEAVES_TRAIN_ACCEL（auto|webgpu|born_cpu|cpu）。
func AccelModeFromEnv() string {
	switch strings.ToLower(strings.TrimSpace(os.Getenv("LEAVES_TRAIN_ACCEL"))) {
	case AccelModeWebGPU, AccelModeBornCPU, AccelModeCPU:
		return strings.ToLower(strings.TrimSpace(os.Getenv("LEAVES_TRAIN_ACCEL")))
	default:
		return AccelModeAuto
	}
}

func effectiveAccelMode(cfg Config) string {
	if cfg.AccelMode != "" {
		return cfg.AccelMode
	}
	return AccelModeFromEnv()
}
