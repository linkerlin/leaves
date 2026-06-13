package leaves

import (
	"fmt"
	"path/filepath"
	"testing"

	"github.com/dmitryikh/leaves/mat"
	"github.com/dmitryikh/leaves/tree"
)

// Phase 1 benchmark 门禁：Native / BornCPU / BornGPU 对比（batch=1/16/256）。
func BenchmarkBreastCancerBackend(b *testing.B) {
	modelPath := filepath.Join("testdata", "lg_breast_cancer.txt")
	dataPath := filepath.Join("testdata", "lg_breast_cancer_data.txt")

	model, err := LGEnsembleFromFile(modelPath, false)
	if err != nil {
		b.Fatalf("load model: %v", err)
	}
	testMat, err := mat.DenseMatFromCsvFile(dataPath, 0, false, " ", 0.0)
	if err != nil {
		b.Fatalf("load data: %v", err)
	}

	backends := []struct {
		name    string
		backend tree.Backend
	}{
		{"Native", tree.BackendNative},
		{"BornCPU", tree.BackendBornCPU},
		{"BornGPU", tree.BackendBornGPU},
	}
	batches := []int{1, 16, 256}

	for _, batch := range batches {
		if batch > testMat.Rows {
			continue
		}
		cols := testMat.Cols
		vals := testMat.Values[:batch*cols]
		preds := make([]float64, batch)

		for _, be := range backends {
			b.Run(fmt.Sprintf("batch=%d/%s", batch, be.name), func(b *testing.B) {
				eng, err := NewEngineFromEnsembleWithOptions(model, &EngineOptions{Backend: be.backend})
				if err != nil {
					b.Fatalf("engine: %v", err)
				}
				defer eng.Close()

				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					if err := eng.PredictDense(vals, batch, cols, preds, 0); err != nil {
						b.Fatal(err)
					}
				}
			})
		}
	}
}

// BenchmarkBreastCancerDelegated 根包委托路径（model.Ensemble 代理）。
func BenchmarkBreastCancerDelegated(b *testing.B) {
	modelPath := filepath.Join("testdata", "lg_breast_cancer.txt")
	dataPath := filepath.Join("testdata", "lg_breast_cancer_data.txt")

	ens, err := LGEnsembleFromFile(modelPath, false)
	if err != nil {
		b.Fatalf("load model: %v", err)
	}
	ens.engineOpts = DefaultEngineOptions()

	testMat, err := mat.DenseMatFromCsvFile(dataPath, 0, false, " ", 0.0)
	if err != nil {
		b.Fatalf("load data: %v", err)
	}

	batch := 16
	cols := testMat.Cols
	vals := testMat.Values[:batch*cols]
	preds := make([]float64, batch)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := ens.PredictDense(vals, batch, cols, preds, 0, 0); err != nil {
			b.Fatal(err)
		}
	}
}
