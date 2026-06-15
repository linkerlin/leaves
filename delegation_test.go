package leaves

import (
	"math"
	"path/filepath"
	"testing"

	"github.com/linkerlin/leaves/mat"
	"github.com/linkerlin/leaves/tree"
)

// TestEnsembleDelegatesToModel 根包 Predict* 经 model.Ensemble 代理后与旧实现一致。
func TestEnsembleDelegatesToModel(t *testing.T) {
	modelPath := filepath.Join("testdata", "lg_breast_cancer.txt")
	dataPath := filepath.Join("testdata", "lg_breast_cancer_data.txt")

	ens, err := LGEnsembleFromFile(modelPath, false)
	if err != nil {
		t.Fatal(err)
	}
	testMat, err := mat.DenseMatFromCsvFile(dataPath, 0, false, " ", 0.0)
	if err != nil {
		t.Fatal(err)
	}

	// 默认委托开启后，PredictSingle 与显式 WithEngineOptions 应一致。
	explicit := ens.WithEngineOptions(DefaultEngineOptions())
	fvals := testMat.Values[:testMat.Cols]
	got := ens.PredictSingle(fvals, 0)
	want := explicit.PredictSingle(fvals, 0)
	if math.Abs(got-want) > 1e-9 {
		t.Errorf("default delegate=%v explicit=%v", got, want)
	}
}

func TestEnsembleDelegateLeafIndex(t *testing.T) {
	modelPath := filepath.Join("testdata", "lg_breast_cancer.txt")
	dataPath := filepath.Join("testdata", "lg_breast_cancer_data.txt")
	predLeavesPath := filepath.Join("testdata", "lg_breast_cancer_data_pred_leaves.txt")

	ens, err := LGEnsembleFromFile(modelPath, false)
	if err != nil {
		t.Fatal(err)
	}
	ens = ens.EnsembleWithLeafPredictions()

	testMat, err := mat.DenseMatFromCsvFile(dataPath, 0, false, " ", 0.0)
	if err != nil {
		t.Fatal(err)
	}
	wantMat, err := mat.DenseMatFromCsvFile(predLeavesPath, 0, false, " ", 0.0)
	if err != nil {
		t.Fatal(err)
	}

	got := make([]float64, testMat.Rows*ens.NOutputGroups())
	if err := ens.PredictDense(testMat.Values, testMat.Rows, testMat.Cols, got, 0, 0); err != nil {
		t.Fatal(err)
	}
	for row := 0; row < testMat.Rows; row++ {
		for col := 0; col < wantMat.Cols; col++ {
			g := got[row*wantMat.Cols+col]
			w := wantMat.Values[row*wantMat.Cols+col]
			if math.Abs(g-w) > 1e-6 {
				t.Fatalf("leaf row=%d col=%d: got %v want %v", row, col, g, w)
			}
		}
	}
}

func TestDelegationCSRMultiThread(t *testing.T) {
	testPath := filepath.Join("testdata", "agaricus_test.libsvm")
	modelPath := filepath.Join("testdata", "xgagaricus.model")
	csr, err := mat.CSRMatFromLibsvmFile(testPath, 0, true)
	if err != nil {
		t.Skip(err)
	}
	legacy, err := XGEnsembleFromFile(modelPath, true)
	if err != nil {
		t.Fatal(err)
	}
	delegated := legacy.WithEngineOptions(DefaultEngineOptions())
	for _, nThreads := range []int{1, 2, 4} {
		legacyPred := make([]float64, csr.Rows())
		delegPred := make([]float64, csr.Rows())
		_ = legacy.PredictCSR(csr.RowHeaders, csr.ColIndexes, csr.Values, legacyPred, 0, nThreads)
		_ = delegated.PredictCSR(csr.RowHeaders, csr.ColIndexes, csr.Values, delegPred, 0, nThreads)
		for i := range legacyPred {
			if math.Abs(legacyPred[i]-delegPred[i]) > 1e-6 {
				t.Fatalf("nThreads=%d row %d: legacy=%f delegated=%f", nThreads, i, legacyPred[i], delegPred[i])
			}
		}
	}
}

func TestWithEngineOptionsSimpleGo(t *testing.T) {
	modelPath := filepath.Join("testdata", "lg_breast_cancer.txt")
	dataPath := filepath.Join("testdata", "lg_breast_cancer_data.txt")

	ens, err := LGEnsembleFromFile(modelPath, false)
	if err != nil {
		t.Fatal(err)
	}
	testMat, err := mat.DenseMatFromCsvFile(dataPath, 0, false, " ", 0.0)
	if err != nil {
		t.Fatal(err)
	}

	native := ens.WithEngineOptions(&EngineOptions{Backend: tree.BackendNative})
	simple := ens.WithEngineOptions(&EngineOptions{Backend: tree.BackendBornCPU})

	outN := make([]float64, testMat.Rows)
	outS := make([]float64, testMat.Rows)
	_ = native.PredictDense(testMat.Values, testMat.Rows, testMat.Cols, outN, 0, 0)
	_ = simple.PredictDense(testMat.Values, testMat.Rows, testMat.Cols, outS, 0, 0)

	for i := range outN {
		if math.Abs(outN[i]-outS[i]) > 1e-5 {
			t.Errorf("row %d: native=%v simplego=%v", i, outN[i], outS[i])
		}
	}
}
