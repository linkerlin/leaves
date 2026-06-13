package io_test

import (
	"path/filepath"
	"testing"

	"github.com/dmitryikh/leaves/io"
)

func TestDetectLightGBMMulticlassModel(t *testing.T) {
	path := filepath.Join("..", "testdata", "lgmulticlass.model")
	format, err := io.DetectFormat(path)
	if err != nil {
		t.Fatal(err)
	}
	if format != io.FormatLightGBM {
		t.Fatalf("expected LightGBM, got %v", format)
	}
}
