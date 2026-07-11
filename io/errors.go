package io

import (
	"errors"
	"fmt"
)

// ErrFormatNotImplemented 表示格式检测成功但加载器尚未实现。
type ErrFormatNotImplemented string

func (e ErrFormatNotImplemented) Error() string {
	return string(e)
}

// IsNotImplemented 判断是否为未实现格式错误（含 ONNX 子集外失败）。
func IsNotImplemented(err error) bool {
	if err == nil {
		return false
	}
	if _, ok := err.(ErrFormatNotImplemented); ok {
		return true
	}
	if errors.Is(err, ErrONNXNotImplemented) {
		return true
	}
	var le *LoadError
	// 占位等级仍算未实现；ONNX 实验路径仅在 cause 为 ErrONNXNotImplemented 时算
	if errors.As(err, &le) && le.Level == SupportPlaceholder {
		return true
	}
	return false
}

// LoadError 可操作的加载/探测失败（Phase D）：含格式等级与下一步 hint。
type LoadError struct {
	Path   string
	Format Format
	Level  SupportLevel
	Op     string // detect | load
	Msg    string
	Hint   string
	cause  error
}

func (e *LoadError) Error() string {
	if e == nil {
		return ""
	}
	name := FormatName(e.Format)
	level := SupportLevelName(e.Level)
	base := e.Msg
	if base == "" && e.cause != nil {
		base = e.cause.Error()
	}
	if base == "" {
		base = "load failed"
	}
	s := fmt.Sprintf("io: %s %s [%s/%s]: %s", e.Op, e.Path, name, level, base)
	if e.Hint != "" {
		s += "\nhint: " + e.Hint
	}
	return s
}

func (e *LoadError) Unwrap() error { return e.cause }

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
	case FormatLeavesJSON:
		return "leaves.json"
	case FormatONNX:
		return "ONNX"
	default:
		return fmt.Sprintf("unknown(%d)", f)
	}
}

// wrapDetectError 将探测失败包装为可操作错误。
func wrapDetectError(path string, err error) error {
	if err == nil {
		return nil
	}
	var le *LoadError
	if errors.As(err, &le) {
		return err
	}
	sup := SupportOf(FormatUnknown)
	return &LoadError{
		Path:   path,
		Format: FormatUnknown,
		Level:  SupportUnsupported,
		Op:     "detect",
		Msg:    err.Error(),
		Hint:   sup.Hint,
		cause:  err,
	}
}

// wrapLoadError 将加载失败包装为带支持等级的错误。
func wrapLoadError(path string, format Format, err error) error {
	if err == nil {
		return nil
	}
	var le *LoadError
	if errors.As(err, &le) {
		return err
	}
	if errors.Is(err, ErrONNXNotImplemented) {
		sup := SupportOf(FormatONNX)
		return &LoadError{
			Path:   path,
			Format: FormatONNX,
			Level:  SupportPlaceholder,
			Op:     "load",
			Msg:    err.Error(),
			Hint:   sup.Hint,
			cause:  err,
		}
	}
	sup := SupportOf(format)
	msg := err.Error()
	hint := sup.Hint
	if format == FormatSklearn {
		hint = "scikit-learn 为实验性支持（窄协议）；" + hint
	}
	return &LoadError{
		Path:   path,
		Format: format,
		Level:  sup.Level,
		Op:     "load",
		Msg:    msg,
		Hint:   hint,
		cause:  err,
	}
}

// newPlaceholderError 占位格式（如 ONNX）的标准错误。
func newPlaceholderError(path string, f Format) error {
	sup := SupportOf(f)
	return &LoadError{
		Path:   path,
		Format: f,
		Level:  SupportPlaceholder,
		Op:     "load",
		Msg:    fmt.Sprintf("%s import not implemented", sup.Name),
		Hint:   sup.Hint,
		cause:  ErrONNXNotImplemented,
	}
}
