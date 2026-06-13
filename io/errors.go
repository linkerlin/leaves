package io

import "fmt"

// ErrFormatNotImplemented 表示格式检测成功但加载器尚未实现。
type ErrFormatNotImplemented string

func (e ErrFormatNotImplemented) Error() string {
	return string(e)
}

// IsNotImplemented 判断是否为未实现格式错误。
func IsNotImplemented(err error) bool {
	_, ok := err.(ErrFormatNotImplemented)
	return ok
}

// FormatName 返回格式的人类可读名称。
func FormatName(f Format) string {
	switch f {
	case FormatLightGBM:
		return "LightGBM text"
	case FormatLightGBMJSON:
		return "LightGBM JSON"
	case FormatXGBoost:
		return "XGBoost binary"
	case FormatXGBoostJSON:
		return "XGBoost JSON"
	case FormatXGBoostUBJSON:
		return "XGBoost UBJSON"
	case FormatSklearn:
		return "scikit-learn pickle"
	default:
		return fmt.Sprintf("unknown(%d)", f)
	}
}
