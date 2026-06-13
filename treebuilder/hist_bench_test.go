package treebuilder

import (
	"testing"

	"github.com/dmitryikh/leaves/data"
)

func benchHistDataset(rows, cols int) (data.Matrix, []int, []float64, []float64, Config) {
	dm, idx, grad, hess := synthHistDataset(rows, cols)
	cfg := Config{
		MaxDepth:      5,
		LearningRate:  0.3,
		Lambda:        1.0,
		MaxBin:        64,
		Gamma:         0,
		HistBinPolicy: HistBinGlobal,
		GlobalBins:    BuildGlobalHistBins(dm, 64, nil),
	}
	return dm, idx, grad, hess, cfg
}

func BenchmarkBuildHistSmall(b *testing.B) {
	dm, idx, grad, hess, cfg := benchHistDataset(500, 8)
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		BuildHist(dm, idx, grad, hess, cfg)
	}
}

func BenchmarkBuildHistMedium(b *testing.B) {
	dm, idx, grad, hess, cfg := benchHistDataset(5000, 32)
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		BuildHist(dm, idx, grad, hess, cfg)
	}
}

func BenchmarkBuildHistLarge(b *testing.B) {
	dm, idx, grad, hess, cfg := benchHistDataset(20000, 50)
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		BuildHist(dm, idx, grad, hess, cfg)
	}
}

func BenchmarkBuildHistPerNodeMedium(b *testing.B) {
	dm, idx, grad, hess, _ := benchHistDataset(5000, 32)
	cfg := Config{
		MaxDepth:      5,
		LearningRate:  0.3,
		Lambda:        1.0,
		MaxBin:        64,
		Gamma:         0,
		HistBinPolicy: HistBinPerNode,
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		BuildHist(dm, idx, grad, hess, cfg)
	}
}
