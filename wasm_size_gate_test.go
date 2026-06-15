package leaves_test

import (
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

// maxWasmBytes CI 门禁上限（含 Go runtime；smoke 模型约 8–12 MiB）。
const maxWasmBytes = 16 << 20

func TestWasmBinarySizeGate(t *testing.T) {
	if testing.Short() {
		t.Skip("wasm size gate skipped in -short")
	}
	switch os.Getenv("LEAVES_WASM_GATE") {
	case "1", "true", "yes":
	default:
		t.Skip("wasm size gate skipped (set LEAVES_WASM_GATE=1; CI wasm job)")
	}

	dir := t.TempDir()
	wasmPath := filepath.Join(dir, "leaves.wasm")
	cmd := exec.Command("go", "build", "-o", wasmPath, "./examples/wasm")
	cmd.Env = append(os.Environ(), "GOOS=js", "GOARCH=wasm")
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("wasm build: %v\n%s", err, out)
	}
	info, err := os.Stat(wasmPath)
	if err != nil {
		t.Fatal(err)
	}
	sz := info.Size()
	t.Logf("leaves.wasm size: %d bytes (%.2f MiB), max %d", sz, float64(sz)/(1<<20), maxWasmBytes)
	if sz > maxWasmBytes {
		t.Errorf("wasm size %d exceeds gate %d", sz, maxWasmBytes)
	}
}
