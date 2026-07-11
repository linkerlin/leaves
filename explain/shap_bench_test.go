package explain_test

import (
	"path/filepath"
	"testing"

	"github.com/linkerlin/leaves/explain"
	"github.com/linkerlin/leaves/io"
)

// BenchmarkTreeSHAPSingle 单样本 Tree SHAP（LIB-22 缓存路径）。
func BenchmarkTreeSHAPSingle(b *testing.B) {
	path := filepath.Join("..", "testdata", "xgboost_smoke.json")
	m, err := io.LoadFromFile(path, &io.LoadOptions{LoadTransformation: false})
	if err != nil {
		b.Skip(err)
	}
	f := m.Forest()
	expl := explain.NewTreeExplainer(f)
	x := make([]float64, m.NFeatures())
	for i := range x {
		x[i] = float64(i%5) * 0.1
	}
	batch := [][]float64{x}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := expl.ShapleyValues(batch); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkTreeSHAPBatch32 32 样本复用 explainer 缓存。
func BenchmarkTreeSHAPBatch32(b *testing.B) {
	path := filepath.Join("..", "testdata", "xgboost_smoke.json")
	m, err := io.LoadFromFile(path, &io.LoadOptions{LoadTransformation: false})
	if err != nil {
		b.Skip(err)
	}
	f := m.Forest()
	expl := explain.NewTreeExplainer(f)
	nf := m.NFeatures()
	batch := make([][]float64, 32)
	for i := range batch {
		x := make([]float64, nf)
		for j := range x {
			x[j] = float64((i+j)%7) * 0.05
		}
		batch[i] = x
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := expl.ShapleyValues(batch); err != nil {
			b.Fatal(err)
		}
	}
}
