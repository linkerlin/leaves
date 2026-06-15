// Package io — ONNX 导入调研（非主路径）。
//
// 结论（2026-06）：
//   - leaves 主路径为原生 LGB/XGB/SK/leaves.json；ONNX 仅适合「已有 ONNX 模型」只读导入。
//   - 推荐方案：外部 onnx2json / onnxruntime 预转换 → XGBoost JSON 或 leaves.json，再 io.LoadFromFile。
//   - 完整 ONNX Graph → ForestIR 需映射 TreeEnsemble/TreeEnsembleRegressor（opset ≥3）或
//     ZipMap+TreeEnsembleClassifier；分类 bitset / categorical 与 leaves 语义需逐 op 对齐。
//   - 依赖：github.com/onnx/onnx-go 可解析 protobuf，但无官方 GBDT 无损 round-trip 保证。
//
// 本文件保留 API 占位，待明确 op 子集后再实现 LoadONNX。
package io

import "fmt"

// ErrONNXNotImplemented ONNX 导入尚未实现。
var ErrONNXNotImplemented = fmt.Errorf("io: onnx import not implemented; convert to xgb json or leaves.json first")

// LoadONNX 从 ONNX 文件加载模型（未实现）。
func LoadONNX(path string, opts *LoadOptions) error {
	_ = path
	_ = opts
	return ErrONNXNotImplemented
}
