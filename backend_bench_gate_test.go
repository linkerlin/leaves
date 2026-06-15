package leaves

import (
	"path/filepath"
	"testing"
	"time"

	"github.com/linkerlin/leaves/mat"
	"github.com/linkerlin/leaves/tree"
)

// TestBenchGateBornCPUSlowerBatch1 CI 门禁：batch=1 时 BornCPU 应显著慢于 Native。
func TestBenchGateBornCPUSlowerBatch1(t *testing.T) {
	if testing.Short() {
		t.Skip("bench gate skipped in -short")
	}
	modelPath := filepath.Join("testdata", "lg_breast_cancer.txt")
	dataPath := filepath.Join("testdata", "lg_breast_cancer_data.txt")

	model, err := LGEnsembleFromFile(modelPath, false)
	if err != nil {
		t.Fatalf("load model: %v", err)
	}
	testMat, err := mat.DenseMatFromCsvFile(dataPath, 0, false, " ", 0.0)
	if err != nil {
		t.Fatalf("load data: %v", err)
	}
	batch := 1
	cols := testMat.Cols
	vals := testMat.Values[:batch*cols]
	preds := make([]float64, batch)

	nativeEng, err := NewEngineFromEnsembleWithOptions(model, &EngineOptions{Backend: tree.BackendNative})
	if err != nil {
		t.Fatal(err)
	}
	defer nativeEng.Close()
	bornEng, err := NewEngineFromEnsembleWithOptions(model, &EngineOptions{Backend: tree.BackendBornCPU})
	if err != nil {
		t.Fatal(err)
	}
	defer bornEng.Close()

	const iters = 200
	nativeDur := timeBench(func() {
		_ = nativeEng.PredictDense(vals, batch, cols, preds, 0)
	}, iters)
	bornDur := timeBench(func() {
		_ = bornEng.PredictDense(vals, batch, cols, preds, 0)
	}, iters)

	ratio := float64(bornDur) / float64(nativeDur)
	const minRatio = 20.0
	if ratio < minRatio {
		t.Errorf("BornCPU/Native batch=1 ratio=%.1fx, want >= %.0fx (native=%v born=%v)", ratio, minRatio, nativeDur, bornDur)
	}
}

func timeBench(fn func(), iters int) time.Duration {
	// warmup
	for i := 0; i < 10; i++ {
		fn()
	}
	start := time.Now()
	for i := 0; i < iters; i++ {
		fn()
	}
	return time.Since(start) / time.Duration(iters)
}
