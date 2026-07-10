// Package io — ONNX 导入策略（Phase D 硬化）。
//
// 策略（稳定承诺）：
//   - 等级：placeholder（SupportPlaceholder）
//   - 不实现 ONNX Graph → ForestIR
//   - 推荐：外部转换 → XGBoost JSON / leaves.json → io.LoadFromFile
//   - 若未来实现，仅考虑极小 TreeEnsemble 子集，且需单独里程碑与回归矩阵
//
// 见 docs/interop-matrix.md 与 SupportOf(FormatONNX)。
package io

import "fmt"

// ErrONNXNotImplemented ONNX 导入尚未实现（可用 errors.Is 判断）。
var ErrONNXNotImplemented = fmt.Errorf("io: onnx import not implemented; convert to xgb json or leaves.json first")

// LoadONNX 从 ONNX 文件加载模型（占位，恒失败）。
func LoadONNX(path string, opts *LoadOptions) error {
	_ = opts
	return newPlaceholderError(path, FormatONNX)
}
