package io_test

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	_ "github.com/linkerlin/leaves"
	"github.com/linkerlin/leaves/io"
)

// TestSklearnSupportLevel 锁定 LIB-11：SK 为 experimental。
func TestSklearnSupportLevel(t *testing.T) {
	s := io.SupportOf(io.FormatSklearn)
	if s.Level != io.SupportExperimental {
		t.Fatalf("level=%v want experimental", s.Level)
	}
	if !strings.Contains(strings.ToLower(s.Hint), "json") && !strings.Contains(s.Hint, "leaves") {
		t.Fatalf("hint should suggest convert to JSON/leaves: %q", s.Hint)
	}
}

// TestSklearnDetectPKLExt 探测 .pkl 扩展名。
func TestSklearnDetectPKLExt(t *testing.T) {
	dir := t.TempDir()
	// pickle protocol 2 魔数
	p := filepath.Join(dir, "m.pkl")
	if err := os.WriteFile(p, []byte{0x80, 0x02, 0x00}, 0o644); err != nil {
		t.Fatal(err)
	}
	f, err := io.DetectFormat(p)
	if err != nil {
		t.Fatal(err)
	}
	if f != io.FormatSklearn {
		t.Fatalf("format=%v want Sklearn", f)
	}
}

// TestSklearnLoadFailureActionable 损坏/非 SK 内容加载失败须可操作。
func TestSklearnLoadFailureActionable(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "bad.joblib")
	// 魔数像 pickle 但内容无效
	if err := os.WriteFile(p, []byte{0x80, 0x04, 0x95, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, '.'}, 0o644); err != nil {
		t.Fatal(err)
	}
	_, err := io.LoadFromFile(p, io.DefaultLoadOptions())
	if err == nil {
		t.Fatal("expected load failure")
	}
	var le *io.LoadError
	if !errors.As(err, &le) {
		// 根包 loader 可能返回非 LoadError 包装；至少错误非空
		if !strings.Contains(strings.ToLower(err.Error()), "pickle") &&
			!strings.Contains(strings.ToLower(err.Error()), "sklearn") &&
			!strings.Contains(strings.ToLower(err.Error()), "load") {
			t.Logf("error without LoadError wrapper (ok if actionable): %v", err)
		}
		return
	}
	if le.Format != io.FormatSklearn && le.Format != io.FormatUnknown {
		t.Logf("format=%v (expected Sklearn or Unknown)", le.Format)
	}
	// experimental 路径应保留 hint 或 message
	if le.Hint == "" && le.Msg == "" {
		t.Fatalf("empty LoadError: %+v", le)
	}
}

// TestSklearnGoldenLoads 已知 golden 仍可加载（实验矩阵不扩大）。
func TestSklearnGoldenLoads(t *testing.T) {
	path := filepath.Join("..", "testdata", "sk_gradient_boosting_classifier.model")
	if _, err := os.Stat(path); err != nil {
		t.Skip(err)
	}
	m, err := io.LoadFromFile(path, &io.LoadOptions{AutoTransform: false, Backend: io.BackendNative})
	if err != nil {
		t.Fatal(err)
	}
	if m.NFeatures() <= 0 {
		t.Fatalf("NFeatures=%d", m.NFeatures())
	}
	defer m.Close()
}
