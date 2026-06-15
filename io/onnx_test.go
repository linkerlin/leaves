package io

import "testing"

func TestLoadONNXNotImplemented(t *testing.T) {
	err := LoadONNX("model.onnx", DefaultLoadOptions())
	if err != ErrONNXNotImplemented {
		t.Fatalf("got %v", err)
	}
}
