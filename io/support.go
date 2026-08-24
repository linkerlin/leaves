package io

import "fmt"

// SupportLevel 互操作支持等级（演进计划 Phase D）。
type SupportLevel int

const (
	// SupportStable 承诺回归维护；进入 testdata 矩阵与 CI。
	SupportStable SupportLevel = iota
	// SupportExperimental 可用但协议窄 / 不保证全版本 SK；失败时给明确边界。
	SupportExperimental
	// SupportPlaceholder API 占位，加载必失败并提示转换路径。
	SupportPlaceholder
	// SupportUnsupported 明确不做或无法识别。
	SupportUnsupported
)

// SupportLevelName 稳定字符串（文档 / JSON）。
func SupportLevelName(l SupportLevel) string {
	switch l {
	case SupportStable:
		return "stable"
	case SupportExperimental:
		return "experimental"
	case SupportPlaceholder:
		return "placeholder"
	case SupportUnsupported:
		return "unsupported"
	default:
		return fmt.Sprintf("level_%d", int(l))
	}
}

// FormatSupport 单格式支持说明。
type FormatSupport struct {
	Format  Format
	Level   SupportLevel
	Name    string // 人类可读
	Summary string // 一句话能力
	Hint    string // 失败时下一步
	// Matrix 回归矩阵锚点（docs/testdata-matrix.md 相关行）
	Matrix string
}

// SupportOf 返回格式的支持等级与操作提示。
func SupportOf(f Format) FormatSupport {
	switch f {
	case FormatLeavesJSON:
		return FormatSupport{
			Format:  f,
			Level:   SupportStable,
			Name:    "leaves.json",
			Summary: "训练主产物；Load/Save 对称；推荐部署格式",
			Hint:    "确认 leaves_version=1 且含 gbtree/dart/gblinear 数据",
			Matrix:  "InferObjective + SaveTrainModel / train/load_test.go",
		}
	case FormatXGBoostJSON:
		return FormatSupport{
			Format:  f,
			Level:   SupportStable,
			Name:    "XGBoost JSON",
			Summary: "XGB 3.x save_model 默认；gbtree/dart/gblinear/多类/排序等",
			Hint:    "用 xgb.save_model('m.json')；logistic 的 base_score 会转 margin",
			Matrix:  "xgboost_smoke.json 等 → TestBornParityFormatMatrix / io/xgb_*",
		}
	case FormatXGBoostUBJSON:
		return FormatSupport{
			Format:  f,
			Level:   SupportStable,
			Name:    "XGBoost UBJSON",
			Summary: "JSON 的二进制等价；预测与 JSON 路径一致",
			Hint:    "扩展名 .ubj 或内容以 { 开头的 UBJSON",
			Matrix:  "xgboost_smoke.ubj → TestBornParityFormatMatrix",
		}
	case FormatXGBoost:
		return FormatSupport{
			Format:  f,
			Level:   SupportStable,
			Name:    "XGBoost binary",
			Summary: "经典 Booster 二进制（binf 或 header 探测）",
			Hint:    "优先改用 XGB JSON/UBJ 便于互操作；二进制无魔数时靠 header 嗅探",
			Matrix:  "xgagaricus.model → io/xgb_bin + parity",
		}
	case FormatLightGBM:
		return FormatSupport{
			Format:  f,
			Level:   SupportStable,
			Name:    "LightGBM text",
			Summary: "text model（tree= / version=）",
			Hint:    "确认是 LGB 导出而非数值 TSV 误用 .txt",
			Matrix:  "lg_breast_cancer.txt → TestBornParityFormatMatrix",
		}
	case FormatLightGBMJSON:
		return FormatSupport{
			Format:  f,
			Level:   SupportStable,
			Name:    "LightGBM JSON",
			Summary: "LGB JSON（tree_info）",
			Hint:    "与 text 语义对齐的 JSON 导出",
			Matrix:  "lg_dart_breast_cancer.json → parity",
		}
	case FormatSklearn:
		return FormatSupport{
			Format:  f,
			Level:   SupportExperimental,
			Name:    "scikit-learn pickle",
			Summary: "实验性：仅部分 GradientBoosting / 历史 pickle 协议；非全 SK 版本矩阵",
			Hint:    "优先导出为 XGBoost JSON 或 leaves.json 再加载；SK 仅窄协议",
			Matrix:  "sk_*.model → TestBornParityFormatMatrix（实验）",
		}
	case FormatONNX:
		return FormatSupport{
			Format:  f,
			Level:   SupportExperimental,
			Name:    "ONNX",
			Summary: "实验：TreeEnsembleRegressor（BRANCH_LEQ/SUM/NONE）与 Classifier（SUM/AVERAGE × NONE/SOFTMAX/LOGISTIC-二类）；完整 Graph 走 LoadOnnxGraph",
			Hint:    "复杂 ONNX 用 LoadOnnxGraph（非 wasm）或转 XGB JSON / leaves.json；子集见 docs/interop-matrix.md",
			Matrix:  "io/onnx_test.go TestONNXTreeEnsembleStump · io/onnx_classifier_test.go",
		}
	default:
		return FormatSupport{
			Format:  FormatUnknown,
			Level:   SupportUnsupported,
			Name:    "unknown",
			Summary: "无法识别的模型文件",
			Hint:    "支持：leaves.json / XGB json|ubj|bin / LGB text|json；SK pickle 实验性；ONNX 请先转换",
			Matrix:  "io/load_detect_test.go",
		}
	}
}

// SupportTable 返回文档/测试用的全表（稳定顺序）。
func SupportTable() []FormatSupport {
	return []FormatSupport{
		SupportOf(FormatLeavesJSON),
		SupportOf(FormatXGBoostJSON),
		SupportOf(FormatXGBoostUBJSON),
		SupportOf(FormatXGBoost),
		SupportOf(FormatLightGBM),
		SupportOf(FormatLightGBMJSON),
		SupportOf(FormatSklearn),
		SupportOf(FormatONNX),
	}
}
