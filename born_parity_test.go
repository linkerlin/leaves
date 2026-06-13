package leaves

import (
	"fmt"
	"math"
	"path/filepath"
	"testing"

	"github.com/dmitryikh/leaves/mat"
	leafmodel "github.com/dmitryikh/leaves/model"
	"github.com/dmitryikh/leaves/tree"
	"github.com/dmitryikh/leaves/util"
)

const bornParityTol = 1e-5

func TestBornParityBreastCancer(t *testing.T) {
	compareEnginesOnFile(t,
		filepath.Join("testdata", "lg_breast_cancer.txt"),
		filepath.Join("testdata", "lg_breast_cancer_data.txt"),
		" ", 0,
	)
}

func TestBornParityNEstimators(t *testing.T) {
	modelPath := filepath.Join("testdata", "lg_breast_cancer.txt")
	dataPath := filepath.Join("testdata", "lg_breast_cancer_data.txt")

	model, err := LGEnsembleFromFile(modelPath, false)
	if err != nil {
		t.Fatal(err)
	}
	testMat, err := mat.DenseMatFromCsvFile(dataPath, 0, false, " ", 0.0)
	if err != nil {
		t.Fatal(err)
	}

	for _, nEst := range []int{1, 5, 10, 50} {
		native, err := NewEngineFromEnsembleWithOptions(model, &EngineOptions{Backend: tree.BackendNative})
		if err != nil {
			t.Fatal(err)
		}
	bornEng, err := NewEngineFromEnsembleWithOptions(model, &EngineOptions{Backend: tree.BackendBornCPU})
		if err != nil {
			t.Fatal(err)
		}
		outN := make([]float64, testMat.Rows)
		outG := make([]float64, testMat.Rows)
		_ = native.PredictDense(testMat.Values, testMat.Rows, testMat.Cols, outN, nEst)
		_ = bornEng.PredictDense(testMat.Values, testMat.Rows, testMat.Cols, outG, nEst)
		native.Close()
		bornEng.Close()
		assertSlicesClose(t, outN, outG, bornParityTol, fmt.Sprintf("nEst=%d", nEst))
	}
}

func TestBornParityKDDCup99(t *testing.T) {
	// kddcup99 含 one-hot + 大 bitset 分类分裂，BornCPU 不支持 CatSmall，回退 Native。
	compareEnginesOnFileWithBackend(t,
		filepath.Join("testdata", "lg_kddcup99.model"),
		filepath.Join("testdata", "kddcup99_test.tsv"),
		"\t", 0,
		tree.BackendBornCPU,
	)
}

func TestBornParityXGBoost(t *testing.T) {
	modelPath := filepath.Join("testdata", "xgagaricus.model")
	model, err := XGEnsembleFromFile(modelPath, false)
	if err != nil {
		t.Fatal(err)
	}

	native, err := NewEngineFromEnsembleWithOptions(model, &EngineOptions{Backend: tree.BackendNative})
	if err != nil {
		t.Fatal(err)
	}
	bornEng, err := NewEngineFromEnsembleWithOptions(model, &EngineOptions{Backend: tree.BackendBornCPU})
	if err != nil {
		t.Fatal(err)
	}
	defer native.Close()
	defer bornEng.Close()

	nFeatures := native.NFeatures()
	rows := 64
	vals := make([]float64, rows*nFeatures)
	outN := make([]float64, rows)
	outG := make([]float64, rows)
	_ = native.PredictDense(vals, rows, nFeatures, outN, 0)
	_ = bornEng.PredictDense(vals, rows, nFeatures, outG, 0)
	assertSlicesClose(t, outN, outG, bornParityTol, "xgboost batch")
}

func TestBornParityLeafIndices(t *testing.T) {
	modelPath := filepath.Join("testdata", "lg_breast_cancer.txt")
	dataPath := filepath.Join("testdata", "lg_breast_cancer_data.txt")
	leavesPath := filepath.Join("testdata", "lg_breast_cancer_data_pred_leaves.txt")

	model, err := LGEnsembleFromFile(modelPath, false)
	if err != nil {
		t.Fatal(err)
	}

	testMat, err := mat.DenseMatFromCsvFile(dataPath, 0, false, " ", 0.0)
	if err != nil {
		t.Fatal(err)
	}
	truth, err := mat.DenseMatFromCsvFile(leavesPath, 0, false, " ", 0.0)
	if err != nil {
		t.Fatal(err)
	}

	nOut := model.NRawOutputGroups() * model.NEstimators()
	ir := EnsembleToModelIR(model)
	if ir == nil || ir.Forest == nil {
		t.Fatal("nil ModelIR")
	}
	ir.NOutputGroups = nOut

	nativeEns, err := leafmodel.NewEnsembleFromIR(ir, tree.ApplyTransformRaw, tree.TransformLeafIndex, tree.BackendNative)
	if err != nil {
		t.Fatal(err)
	}
	bornEngEns, err := leafmodel.NewEnsembleFromIR(ir, tree.ApplyTransformRaw, tree.TransformLeafIndex, tree.BackendBornCPU)
	if err != nil {
		t.Fatalf("bornEng engine: %v", err)
	}
	defer nativeEns.Close()
	defer bornEngEns.Close()

	outN := make([]float64, testMat.Rows*nOut)
	outG := make([]float64, testMat.Rows*nOut)
	if err := nativeEns.Engine().PredictLeafIndicesDense(testMat.Values, testMat.Rows, testMat.Cols, outN); err != nil {
		t.Fatal(err)
	}
	if err := bornEngEns.Engine().PredictLeafIndicesDense(testMat.Values, testMat.Rows, testMat.Cols, outG); err != nil {
		t.Fatal(err)
	}
	assertSlicesClose(t, outN, outG, 0, "leaf native vs bornEng")
	if err := util.AlmostEqualFloat64Slices(truth.Values, outN, 0); err != nil {
		t.Errorf("native leaf vs truth: %v", err)
	}
}

func compareEnginesOnFile(t *testing.T, modelPath, dataPath, sep string, nEst int) {
	compareEnginesOnFileWithBackend(t, modelPath, dataPath, sep, nEst, tree.BackendBornCPU)
}

func compareEnginesOnFileWithBackend(t *testing.T, modelPath, dataPath, sep string, nEst int, bornEngBackend tree.Backend) {
	t.Helper()
	model, err := LGEnsembleFromFile(modelPath, false)
	if err != nil {
		t.Fatal(err)
	}
	testMat, err := mat.DenseMatFromCsvFile(dataPath, 0, false, sep, 0.0)
	if err != nil {
		t.Fatal(err)
	}

	native, err := NewEngineFromEnsembleWithOptions(model, &EngineOptions{Backend: tree.BackendNative})
	if err != nil {
		t.Fatal(err)
	}
	bornEng, err := NewEngineFromEnsembleWithOptions(model, &EngineOptions{Backend: bornEngBackend})
	if err != nil {
		t.Fatal(err)
	}
	defer native.Close()
	defer bornEng.Close()

	outN := make([]float64, testMat.Rows*native.NOutputGroups())
	outG := make([]float64, testMat.Rows*bornEng.NOutputGroups())
	if err := native.PredictDense(testMat.Values, testMat.Rows, testMat.Cols, outN, nEst); err != nil {
		t.Fatal(err)
	}
	if err := bornEng.PredictDense(testMat.Values, testMat.Rows, testMat.Cols, outG, nEst); err != nil {
		t.Fatal(err)
	}
	assertSlicesClose(t, outN, outG, bornParityTol, modelPath)
}

func assertSlicesClose(t *testing.T, a, b []float64, tol float64, label string) {
	t.Helper()
	if len(a) != len(b) {
		t.Fatalf("%s: length mismatch %d vs %d", label, len(a), len(b))
	}
	maxDiff := 0.0
	for i := range a {
		d := math.Abs(a[i] - b[i])
		if tol == 0 && a[i] != b[i] {
			t.Errorf("%s idx=%d: %f != %f", label, i, a[i], b[i])
		}
		if d > maxDiff {
			maxDiff = d
		}
		if d > tol {
			t.Errorf("%s idx=%d: diff=%e (a=%f b=%f)", label, i, d, a[i], b[i])
		}
	}
	t.Logf("%s: max diff=%e", label, maxDiff)
}

func TestBornSparseZeroInitParity(t *testing.T) {
	modelPath := filepath.Join("testdata", "xgagaricus.model")
	model, err := XGEnsembleFromFile(modelPath, true)
	if err != nil {
		t.Fatal(err)
	}
	testPath := filepath.Join("testdata", "agaricus_test.libsvm")
	csr, err := mat.CSRMatFromLibsvmFile(testPath, 0, true)
	if err != nil {
		t.Skip(err)
	}
	if csr.Rows() < 2 {
		t.Skip("need at least 2 rows")
	}
	nf := model.NFeatures()
	zeroDense := make([]float64, 2*nf)
	for i := 0; i < 2; i++ {
		start := csr.RowHeaders[i]
		end := csr.RowHeaders[i+1]
		for j := start; j < end; j++ {
			if csr.ColIndexes[j] < nf {
				zeroDense[i*nf+csr.ColIndexes[j]] = csr.Values[j]
			}
		}
	}
	native, err := NewEngineFromEnsembleWithOptions(model, &EngineOptions{Backend: tree.BackendNative})
	if err != nil {
		t.Fatal(err)
	}
	bornEng, err := NewEngineFromEnsembleWithOptions(model, &EngineOptions{Backend: tree.BackendBornCPU})
	if err != nil {
		t.Fatal(err)
	}
	defer native.Close()
	defer bornEng.Close()
	outN := make([]float64, 2)
	outG := make([]float64, 2)
	_ = native.PredictDense(zeroDense, 2, nf, outN, 0)
	_ = bornEng.PredictDense(zeroDense, 2, nf, outG, 0)
	assertSlicesClose(t, outN, outG, 1e-4, "xgagaricus_zero_dense2")
}

func TestBornCSRNaNParity(t *testing.T) {
	// 两路稀疏 CSR 行 → 稠密 NaN 底 + 填值，验证 batch 下图遍历与 Native 一致。
	modelPath := filepath.Join("testdata", "xgagaricus.model")
	model, err := XGEnsembleFromFile(modelPath, true)
	if err != nil {
		t.Fatal(err)
	}
	testPath := filepath.Join("testdata", "agaricus_test.libsvm")
	csr, err := mat.CSRMatFromLibsvmFile(testPath, 0, true)
	if err != nil {
		t.Skip(err)
	}
	if csr.Rows() < 2 {
		t.Skip("need at least 2 rows")
	}
	nf := model.NFeatures()
	dense := make([]float64, 2*nf)
	for i := 0; i < 2; i++ {
		row := dense[i*nf : (i+1)*nf]
		for j := range row {
			row[j] = math.NaN()
		}
		start := csr.RowHeaders[i]
		end := csr.RowHeaders[i+1]
		for j := start; j < end; j++ {
			if csr.ColIndexes[j] < nf {
				row[csr.ColIndexes[j]] = csr.Values[j]
			}
		}
	}

	native, err := NewEngineFromEnsembleWithOptions(model, &EngineOptions{Backend: tree.BackendNative})
	if err != nil {
		t.Fatal(err)
	}
	bornEng, err := NewEngineFromEnsembleWithOptions(model, &EngineOptions{Backend: tree.BackendBornCPU})
	if err != nil {
		t.Fatal(err)
	}
	defer native.Close()
	defer bornEng.Close()

	outN := make([]float64, 2)
	outG := make([]float64, 2)
	_ = native.PredictDense(dense, 2, nf, outN, 0)
	_ = bornEng.PredictDense(dense, 2, nf, outG, 0)
	assertSlicesClose(t, outN, outG, 1e-4, "xgagaricus_nan_dense2")
}

// TestBornParityMatrix 门禁：Native vs BornCPU（+ BornGPU 若可用）× batch 规模。
func TestBornParityMatrix(t *testing.T) {
	modelPath := filepath.Join("testdata", "lg_breast_cancer.txt")
	dataPath := filepath.Join("testdata", "lg_breast_cancer_data.txt")

	model, err := LGEnsembleFromFile(modelPath, false)
	if err != nil {
		t.Fatal(err)
	}
	testMat, err := mat.DenseMatFromCsvFile(dataPath, 0, false, " ", 0.0)
	if err != nil {
		t.Fatal(err)
	}

	backends := []struct {
		name    string
		backend tree.Backend
	}{
		{"BornCPU", tree.BackendBornCPU},
		{"BornGPU", tree.BackendBornGPU},
	}
	if !tree.BornWebGPUAvailable() {
		backends = backends[:1]
	}

	batches := []int{1, 16, 256}
	for _, batch := range batches {
		if batch > testMat.Rows {
			continue
		}
		cols := testMat.Cols
		vals := testMat.Values[:batch*cols]

		native, err := NewEngineFromEnsembleWithOptions(model, &EngineOptions{Backend: tree.BackendNative})
		if err != nil {
			t.Fatal(err)
		}
		want := make([]float64, batch)
		if err := native.PredictDense(vals, batch, cols, want, 0); err != nil {
			t.Fatal(err)
		}
		native.Close()

		for _, be := range backends {
			t.Run(fmt.Sprintf("batch=%d/%s", batch, be.name), func(t *testing.T) {
				eng, err := NewEngineFromEnsembleWithOptions(model, &EngineOptions{Backend: be.backend})
				if err != nil {
					t.Fatal(err)
				}
				defer eng.Close()

				got := make([]float64, batch)
				if err := eng.PredictDense(vals, batch, cols, got, 0); err != nil {
					t.Fatal(err)
				}
				assertSlicesClose(t, want, got, bornParityTol, fmt.Sprintf("%s batch=%d", be.name, batch))
			})
		}
	}
}
