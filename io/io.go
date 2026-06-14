// Package io 提供 GBRT 模型的加载函数。
// 支持 LightGBM（文本/JSON 格式）、XGBoost（二进制格式）和 scikit-learn（pickle 格式）。
//
// Phase 0 中此为兼容桥——内部委托给根包的现有 IO 实现。
// Phase 1 起，加载函数将直接生成 tree.ForestIR。
package io

import (
	"github.com/dmitryikh/leaves/model"
	"github.com/dmitryikh/leaves/tree"
)

// Format 模型格式枚举。
type Format int

const (
	// FormatUnknown 未知格式。
	FormatUnknown Format = iota
	// FormatLightGBM TXT 文本格式。
	FormatLightGBM
	// FormatLightGBMJSON JSON 格式。
	FormatLightGBMJSON
	// FormatXGBoost 二进制格式（gbtree / dart / gblinear）。
	FormatXGBoost
	// FormatXGBoostJSON JSON 格式。
	FormatXGBoostJSON
	// FormatXGBoostUBJSON UBJSON 格式（XGBoost 3.x 默认二进制序列化）。
	FormatXGBoostUBJSON
	// FormatSklearn pickle 格式。
	FormatSklearn
	// FormatLeavesJSON leaves 训练产出 JSON。
	FormatLeavesJSON
)

// Backend 推理后端选择。
type Backend = tree.Backend

const (
	BackendNative   = tree.BackendNative
	BackendBornCPU  = tree.BackendBornCPU
	BackendBornGPU  = tree.BackendBornGPU
	BackendAuto     = tree.BackendAuto
)

// WorkloadHint 推理 workload 提示（BackendAuto 时使用）。
type WorkloadHint = tree.WorkloadHint

// DeployTarget 部署场景。
type DeployTarget = tree.DeployTarget

const (
	DeployDefault = tree.DeployDefault
	DeployWASM    = tree.DeployWASM
)

// LoadOptions 模型加载选项。
type LoadOptions struct {
	// LoadTransformation 是否自动检测并加载变换函数。
	LoadTransformation bool
	// Backend 推理后端（BackendAuto 时结合 Workload 自动选择）。
	Backend Backend
	// Workload BackendAuto 时的 workload 提示。
	Workload WorkloadHint
}

// SelectBackend 根据 ModelIR 与 workload 选择后端（供 BackendAuto 使用）。
func SelectBackend(ir *model.ModelIR, hint WorkloadHint) Backend {
	return model.SelectBackend(ir, hint)
}

// DefaultLoadOptions 默认 BackendAuto（小 batch Native，大 batch 可 Born）。
func DefaultLoadOptions() *LoadOptions {
	return &LoadOptions{
		LoadTransformation: false,
		Backend:            BackendAuto,
		Workload:           tree.DefaultWorkloadHint(),
	}
}
