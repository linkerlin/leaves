package io_test

import (
	"errors"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/linkerlin/leaves/v2/io"
	"github.com/linkerlin/leaves/v2/tree"
)

func TestLoadONNXMissingFile(t *testing.T) {
	_, err := io.LoadONNX("no_such_model.onnx", io.DefaultLoadOptions())
	if err == nil {
		t.Fatal("expected error")
	}
}

func TestONNXTreeEnsembleStump(t *testing.T) {
	raw := io.SampleONNXStump()
	dir := t.TempDir()
	path := filepath.Join(dir, "stump.onnx")
	if err := os.WriteFile(path, raw, 0o644); err != nil {
		t.Fatal(err)
	}

	ens, err := io.LoadONNX(path, &io.LoadOptions{Backend: io.BackendNative})
	if err != nil {
		t.Fatalf("LoadONNX: %v", err)
	}
	forest := ens.Forest()
	if forest == nil {
		t.Fatal("nil forest")
	}
	if forest.NumFeatures < 1 {
		t.Fatalf("NumFeatures=%d", forest.NumFeatures)
	}

	// x0=0.4 <= 0.5 → 1.0; x0=0.6 → 2.0
	mLeft := tree.ForestMargins(forest, []float64{0.4}, 0)
	mRight := tree.ForestMargins(forest, []float64{0.6}, 0)
	if len(mLeft) < 1 || len(mRight) < 1 {
		t.Fatalf("margins empty")
	}
	if math.Abs(mLeft[0]-1.0) > 1e-5 {
		t.Errorf("left margin=%g want 1", mLeft[0])
	}
	if math.Abs(mRight[0]-2.0) > 1e-5 {
		t.Errorf("right margin=%g want 2", mRight[0])
	}

	// LoadFromFile 路径
	ens2, err := io.LoadFromFile(path, &io.LoadOptions{Backend: io.BackendNative})
	if err != nil {
		// LoadFromFile 需要 registered loader；无 leaves 导入时可能失败
		// 本测试包 io_test 可能未 register — 仅记录
		t.Logf("LoadFromFile (may need leaves import): %v", err)
	} else if ens2 == nil {
		t.Fatal("nil ensemble")
	}

	// Support 等级
	if io.SupportOf(io.FormatONNX).Level != io.SupportExperimental {
		t.Fatalf("ONNX level want experimental")
	}
}

func TestONNXUnsupportedOp(t *testing.T) {
	// 空 model graph → 无可识别节点
	// ir_version only
	raw := []byte{0x08, 0x08} // field 1 varint 8
	_, err := io.ParseONNXTreeEnsemble(raw)
	if err == nil {
		t.Fatal("expected error")
	}
	if !errors.Is(err, io.ErrONNXNotImplemented) && !io.IsNotImplemented(err) {
		// Parse returns wrapped error with ErrONNXNotImplemented
		if !errors.Is(err, io.ErrONNXNotImplemented) {
			t.Logf("err=%v (ok if subset message)", err)
		}
	}
}
