package io

import (
	"errors"
	"testing"
)

func TestLoadONNXNotImplemented(t *testing.T) {
	err := LoadONNX("model.onnx", DefaultLoadOptions())
	if err == nil {
		t.Fatal("expected error")
	}
	// 新错误为 *LoadError，cause 链上仍可 Is ErrONNXNotImplemented 或 IsNotImplemented
	if !IsNotImplemented(err) {
		t.Fatalf("IsNotImplemented: %v", err)
	}
	var le *LoadError
	if !errors.As(err, &le) {
		t.Fatalf("want *LoadError: %T", err)
	}
	if le.Format != FormatONNX {
		t.Fatalf("format %v", le.Format)
	}
}
