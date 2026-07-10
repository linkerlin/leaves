package io_test

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/linkerlin/leaves/io"
)

func TestSupportTableComplete(t *testing.T) {
	table := io.SupportTable()
	if len(table) < 8 {
		t.Fatalf("table len=%d", len(table))
	}
	seen := map[io.Format]bool{}
	for _, row := range table {
		if row.Name == "" || row.Hint == "" || row.Summary == "" {
			t.Fatalf("incomplete row: %+v", row)
		}
		seen[row.Format] = true
	}
	for _, f := range []io.Format{
		io.FormatLeavesJSON, io.FormatXGBoostJSON, io.FormatXGBoostUBJSON,
		io.FormatXGBoost, io.FormatLightGBM, io.FormatLightGBMJSON,
		io.FormatSklearn, io.FormatONNX,
	} {
		if !seen[f] {
			t.Errorf("missing format %v", f)
		}
	}
	if io.SupportOf(io.FormatSklearn).Level != io.SupportExperimental {
		t.Fatal("sklearn should be experimental")
	}
	if io.SupportOf(io.FormatONNX).Level != io.SupportPlaceholder {
		t.Fatal("onnx should be placeholder")
	}
	if io.SupportOf(io.FormatXGBoostJSON).Level != io.SupportStable {
		t.Fatal("xgb json should be stable")
	}
}

func TestDetectFormatONNX(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "m.onnx")
	if err := os.WriteFile(p, []byte("not-real"), 0o644); err != nil {
		t.Fatal(err)
	}
	f, err := io.DetectFormat(p)
	if err != nil {
		t.Fatal(err)
	}
	if f != io.FormatONNX {
		t.Fatalf("got %v", f)
	}
}

func TestLoadONNXActionableError(t *testing.T) {
	err := io.LoadONNX("model.onnx", io.DefaultLoadOptions())
	if err == nil {
		t.Fatal("expected error")
	}
	var le *io.LoadError
	if !errors.As(err, &le) {
		t.Fatalf("want *LoadError, got %T %v", err, err)
	}
	if le.Format != io.FormatONNX || le.Level != io.SupportPlaceholder {
		t.Fatalf("%+v", le)
	}
	if !strings.Contains(le.Hint, "json") && !strings.Contains(le.Hint, "leaves") {
		t.Fatalf("hint should suggest conversion: %q", le.Hint)
	}
	if !io.IsNotImplemented(err) {
		t.Fatal("IsNotImplemented")
	}
	// LoadFromFile 同路径
	if _, err := io.LoadFromFile(filepath.Join(t.TempDir(), "x.onnx"), io.DefaultLoadOptions()); err == nil {
		// 文件不存在可能先失败；写一个空 onnx
	}
	dir := t.TempDir()
	p := filepath.Join(dir, "x.onnx")
	_ = os.WriteFile(p, []byte("x"), 0o644)
	_, err = io.LoadFromFile(p, io.DefaultLoadOptions())
	if err == nil {
		t.Fatal("LoadFromFile onnx should fail")
	}
	if !errors.As(err, &le) || le.Level != io.SupportPlaceholder {
		t.Fatalf("LoadFromFile: %v", err)
	}
	if !strings.Contains(err.Error(), "hint:") {
		t.Fatalf("error should include hint: %v", err)
	}
}

func TestDetectTabularTxtActionable(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "data.txt")
	if err := os.WriteFile(p, []byte("1.0 2.0 3.0\n4.0 5.0 6.0\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	// space-separated may not trigger CSV heuristic; use comma
	p2 := filepath.Join(dir, "data2.txt")
	if err := os.WriteFile(p2, []byte("1.0,2.0,3.0\n4.0,5.0,6.0\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	_, err := io.DetectFormat(p2)
	if err == nil {
		t.Fatal("expected tabular detect error")
	}
	var le *io.LoadError
	if !errors.As(err, &le) {
		t.Fatalf("want LoadError: %v", err)
	}
	if !strings.Contains(le.Hint, "data.FromFile") && !strings.Contains(le.Hint, "train") {
		t.Fatalf("hint: %q", le.Hint)
	}
}

func TestFormatNameLeavesAndONNX(t *testing.T) {
	if io.FormatName(io.FormatLeavesJSON) != "leaves.json" {
		t.Fatal(io.FormatName(io.FormatLeavesJSON))
	}
	if io.FormatName(io.FormatONNX) != "ONNX" {
		t.Fatal(io.FormatName(io.FormatONNX))
	}
	if io.SupportLevelName(io.SupportStable) != "stable" {
		t.Fatal(io.SupportLevelName(io.SupportStable))
	}
}
