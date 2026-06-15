package leaves

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"

	leafio "github.com/linkerlin/leaves/io"
	"github.com/linkerlin/leaves/mat"
	"github.com/linkerlin/leaves/tree"
)

// formatParityCase 单格式 parity 用例（Native vs BornCPU/GPU）。
type formatParityCase struct {
	name    string
	model   string
	data    string // 空则用零特征 batch
	sep     string
	skipRow int // DenseMatFromCsvFile skip header rows
}

// TestBornParityFormatMatrix testdata 全格式 × batch × Born 后端门禁。
func TestBornParityFormatMatrix(t *testing.T) {
	cases := []formatParityCase{
		{name: "LGBText", model: "testdata/lg_breast_cancer.txt", data: "testdata/lg_breast_cancer_data.txt", sep: " "},
		{name: "LGBJSON", model: "testdata/lg_dart_breast_cancer.json", data: "testdata/breast_cancer_test.tsv", sep: "\t"},
		{name: "LGBModel", model: "testdata/lg_dart_breast_cancer.model", data: "testdata/breast_cancer_test.tsv", sep: "\t"},
		{name: "XGBBin", model: "testdata/xgagaricus.model"},
		{name: "XGBJSON", model: "testdata/xgboost_smoke.json"},
		{name: "XGBUBJ", model: "testdata/xgboost_smoke.ubj"},
		{name: "XGBRFJSON", model: "testdata/xgboost_rf_smoke.json"},
		{name: "XGBCatJSON", model: "testdata/xgboost_categorical_smoke.json"},
		{name: "XGBMultiTarget", model: "testdata/xgboost_multitarget_vector.json"},
		{name: "XGBGblinearJSON", model: "testdata/xgboost_gblinear_smoke.json"},
		{name: "XGBGammaJSON", model: "testdata/xgboost_gamma_smoke.json"},
		{name: "XGBPoissonJSON", model: "testdata/xgboost_poisson_smoke.json"},
		{name: "XGBDartJSON", model: "testdata/xgboost_dart_smoke.json"},
		{name: "XGBMulticlassJSON", model: "testdata/xgboost_multiclass_smoke.json"},
		{name: "SKPickle", model: "testdata/sk_gradient_boosting_classifier.model", data: "testdata/sk_gradient_boosting_classifier_test.libsvm"},
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

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := os.Stat(tc.model); err != nil {
				t.Skipf("missing model %s", tc.model)
			}
			if tc.name == "SKPickle" {
				runSKParityMatrix(t, tc, backends, batches)
				return
			}

			mNative, err := leafio.LoadFromFile(tc.model, &leafio.LoadOptions{
				LoadTransformation: false,
				Backend:            leafio.BackendNative,
			})
			if err != nil {
				t.Fatalf("load native: %v", err)
			}
			defer mNative.Close()

			vals, batch, cols := formatParityBatch(t, tc, mNative.NFeatures(), 256)
			if batch == 0 {
				t.Fatal("empty batch")
			}
			nOut := mNative.NOutputGroups()

			for _, bsize := range batches {
				if bsize > batch {
					continue
				}
				sub := vals[:bsize*cols]
				want := make([]float64, bsize*nOut)
				if err := mNative.PredictDense(sub, bsize, cols, want, 0, 0); err != nil {
					t.Fatalf("native batch=%d: %v", bsize, err)
				}

				for _, be := range backends {
					t.Run(fmt.Sprintf("batch=%d/%s", bsize, be.name), func(t *testing.T) {
						mBorn, err := leafio.LoadFromFile(tc.model, &leafio.LoadOptions{
							LoadTransformation: false,
							Backend:            be.backend,
						})
						if err != nil {
							t.Fatalf("load %s: %v", be.name, err)
						}
						defer mBorn.Close()

						got := make([]float64, bsize*nOut)
						if err := mBorn.PredictDense(sub, bsize, cols, got, 0, 0); err != nil {
							t.Fatalf("predict: %v", err)
						}
						assertSlicesClose(t, want, got, bornParityTol,
							fmt.Sprintf("%s/%s batch=%d", tc.name, be.name, bsize))
					})
				}
			}
		})
	}
}

func formatParityBatch(t *testing.T, tc formatParityCase, nFeat, maxRows int) (vals []float64, rows, cols int) {
	t.Helper()
	if tc.data == "" {
		rows = maxRows
		cols = nFeat
		return make([]float64, rows*cols), rows, cols
	}
	if _, err := os.Stat(tc.data); err != nil {
		t.Skipf("missing data %s", tc.data)
	}
	if filepath.Ext(tc.data) == ".libsvm" {
		csr, err := mat.CSRMatFromLibsvmFile(tc.data, tc.skipRow, true)
		if err != nil {
			t.Fatal(err)
		}
		rows = csr.Rows()
		if rows > maxRows {
			rows = maxRows
		}
		cols = nFeat
		vals = make([]float64, rows*cols)
		for i := 0; i < rows; i++ {
			start := csr.RowHeaders[i]
			end := csr.RowHeaders[i+1]
			row := vals[i*cols : (i+1)*cols]
			for j := start; j < end; j++ {
				c := csr.ColIndexes[j]
				if c < cols {
					row[c] = csr.Values[j]
				}
			}
		}
		return vals, rows, cols
	}
	matData, err := mat.DenseMatFromCsvFile(tc.data, tc.skipRow, false, tc.sep, 0.0)
	if err != nil {
		t.Fatal(err)
	}
	rows = matData.Rows
	if rows > maxRows {
		rows = maxRows
	}
	cols = matData.Cols
	vals = matData.Values[:rows*cols]
	return vals, rows, cols
}

// TestEnsembleDelegatePredictDense 根包 PredictDense 经 model.Ensemble 代理与直连一致。
func TestEnsembleDelegatePredictDense(t *testing.T) {
	modelPath := filepath.Join("testdata", "lg_breast_cancer.txt")
	dataPath := filepath.Join("testdata", "lg_breast_cancer_data.txt")

	legacy, err := LGEnsembleFromFile(modelPath, false)
	if err != nil {
		t.Fatal(err)
	}
	legacy.engineOpts = DefaultEngineOptions()

	direct, err := leafio.LoadFromFile(modelPath, &leafio.LoadOptions{
		LoadTransformation: false,
		Backend:            leafio.BackendNative,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer direct.Close()

	testMat, err := mat.DenseMatFromCsvFile(dataPath, 0, false, " ", 0.0)
	if err != nil {
		t.Fatal(err)
	}
	batch := 16
	if batch > testMat.Rows {
		batch = testMat.Rows
	}
	cols := testMat.Cols
	vals := testMat.Values[:batch*cols]

	want := make([]float64, batch)
	got := make([]float64, batch)
	if err := direct.PredictDense(vals, batch, cols, want, 0, 0); err != nil {
		t.Fatal(err)
	}
	if err := legacy.PredictDense(vals, batch, cols, got, 0, 0); err != nil {
		t.Fatal(err)
	}
	assertSlicesClose(t, want, got, bornParityTol, "delegate vs direct")
}

// runSKParityMatrix SK joblib 模型不经 io.DetectFormat，走遗留 SKEnsembleFromFile。
func runSKParityMatrix(t *testing.T, tc formatParityCase, backends []struct {
	name    string
	backend tree.Backend
}, batches []int) {
	t.Helper()
	model, err := SKEnsembleFromFile(tc.model, false)
	if err != nil {
		t.Skipf("SK load: %v", err)
	}
	vals, batch, cols := formatParityBatch(t, tc, model.NFeatures(), 256)
	if batch == 0 {
		t.Fatal("empty batch")
	}
	nOut := model.NOutputGroups()

	for _, bsize := range batches {
		if bsize > batch {
			continue
		}
		sub := vals[:bsize*cols]
		native, err := NewEngineFromEnsembleWithOptions(model, &EngineOptions{Backend: tree.BackendNative})
		if err != nil {
			t.Fatal(err)
		}
		want := make([]float64, bsize*nOut)
		if err := native.PredictDense(sub, bsize, cols, want, 0); err != nil {
			t.Fatal(err)
		}
		native.Close()

		for _, be := range backends {
			t.Run(fmt.Sprintf("batch=%d/%s", bsize, be.name), func(t *testing.T) {
				eng, err := NewEngineFromEnsembleWithOptions(model, &EngineOptions{Backend: be.backend})
				if err != nil {
					t.Fatal(err)
				}
				defer eng.Close()
				got := make([]float64, bsize*nOut)
				if err := eng.PredictDense(sub, bsize, cols, got, 0); err != nil {
					t.Fatal(err)
				}
				assertSlicesClose(t, want, got, bornParityTol,
					fmt.Sprintf("%s/%s batch=%d", tc.name, be.name, bsize))
			})
		}
	}
}
