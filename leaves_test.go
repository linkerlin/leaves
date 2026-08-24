package leaves

import (
	"bufio"
	"os"
	"path/filepath"
	"testing"

	"github.com/linkerlin/leaves/v2/io"
	"github.com/linkerlin/leaves/v2/mat"
	"github.com/linkerlin/leaves/v2/transformation"
	"github.com/linkerlin/leaves/v2/util"
)

func isFileExists(filename string) bool {
	f, err := os.Open(filename)
	if err != nil {
		return false
	}
	f.Close()
	return true
}

func skipTestIfFileNotExist(t *testing.T, filenames ...string) {
	for _, filename := range filenames {
		if !isFileExists(filename) {
			t.Skipf("Skipping due to absence of  file: %s", filename)
		}
	}
}

func skipBenchmarkIfFileNotExist(t *testing.B, filenames ...string) {
	for _, filename := range filenames {
		if !isFileExists(filename) {
			t.Skipf("Skipping due to absence of  file: %s", filename)
		}
	}
}

func TestXGAgaricus(t *testing.T) {
	InnerTestXGAgaricus(t, 1)
	InnerTestXGAgaricus(t, 2)
	InnerTestXGAgaricus(t, 3)
	InnerTestXGAgaricus(t, 4)
}

func InnerTestXGAgaricus(t *testing.T, nThreads int) {
	// loading test data
	testPath := filepath.Join("testdata", "agaricus_test.libsvm")
	modelPath := filepath.Join("testdata", "xgagaricus.model")
	truePath := filepath.Join("testdata", "xgagaricus_true_predictions.txt")
	skipTestIfFileNotExist(t, testPath, modelPath, truePath)
	csr, err := mat.CSRMatFromLibsvmFile(testPath, 0, true)
	if err != nil {
		t.Fatal(err)
	}

	// loading model
	model, err := XGEnsembleFromFile(modelPath, true)
	if err != nil {
		t.Fatal(err)
	}
	if model.NEstimators() != 3 {
		t.Fatalf("expected 3 trees (got %d)", model.NEstimators())
	}
	if model.NOutputGroups() != 1 {
		t.Fatalf("expected NOutputGroups = 1 (got %d)", model.NOutputGroups())
	}
	if model.NRawOutputGroups() != 1 {
		t.Fatalf("expected NRawOutputGroups = 1 (got %d)", model.NRawOutputGroups())
	}
	if model.Transformation().Type() != transformation.Logistic {
		t.Fatalf("expected TransforType = Logistic (got %s)", model.Transformation().Name())
	}

	// loading true predictions as DenseMat
	truePredictions, err := mat.DenseMatFromCsvFile(truePath, 0, false, ",", 0.0)
	if err != nil {
		t.Fatal(err)
	}

	// do predictions with transformation inside
	predictions := make([]float64, csr.Rows()*model.NOutputGroups())
	model.PredictCSR(csr.RowHeaders, csr.ColIndexes, csr.Values, predictions, 0, nThreads)
	// compare results
	if err := util.AlmostEqualFloat64Slices(truePredictions.Values, predictions, 1e-7); err != nil {
		t.Fatalf("different predictions: %s", err.Error())
	}

	// do raw predictions with transformation outside
	rawModel := model.EnsembleWithRawPredictions()
	rawModel.PredictCSR(csr.RowHeaders, csr.ColIndexes, csr.Values, predictions, 0, nThreads)
	util.SigmoidFloat64SliceInplace(predictions)
	// compare results
	if err := util.AlmostEqualFloat64Slices(truePredictions.Values, predictions, 1e-7); err != nil {
		t.Fatalf("different predictions: %s", err.Error())
	}
}

func TestXGBLinAgaricus(t *testing.T) {
	InnerTestXGBLinAgaricus(t, true, 1)
	InnerTestXGBLinAgaricus(t, true, 2)
	InnerTestXGBLinAgaricus(t, true, 3)
	InnerTestXGBLinAgaricus(t, true, 4)
	InnerTestXGBLinAgaricus(t, false, 1)
	InnerTestXGBLinAgaricus(t, false, 2)
	InnerTestXGBLinAgaricus(t, false, 3)
	InnerTestXGBLinAgaricus(t, false, 4)
}

func InnerTestXGBLinAgaricus(t *testing.T, loadTransformation bool, nThreads int) {
	testPath := filepath.Join("testdata", "agaricus_test.libsvm")
	modelPath := filepath.Join("testdata", "xgblin_agaricus.model")
	truePath := filepath.Join("testdata", "xgblin_agaricus_true_raw_predictions.txt")
	if loadTransformation {
		truePath = filepath.Join("testdata", "xgblin_agaricus_true_predictions.txt")
	}

	// loading test data
	csr, err := mat.CSRMatFromLibsvmFile(testPath, 0, true)
	if err != nil {
		t.Fatal(err)
	}

	// loading model with or withoud transformation function (depends on loadTransformation)
	model, err := XGBLinearFromFile(modelPath, loadTransformation)
	if err != nil {
		t.Fatal(err)
	}

	// loading true predictions as DenseMat
	truePredictions, err := mat.DenseMatFromCsvFile(truePath, 0, false, ",", 0.0)
	if err != nil {
		t.Fatal(err)
	}

	// do predictions
	predictions := make([]float64, csr.Rows())
	model.PredictCSR(csr.RowHeaders, csr.ColIndexes, csr.Values, predictions, 0, nThreads)
	// compare results
	if err := util.AlmostEqualFloat64Slices(truePredictions.Values, predictions, 1e-5); err != nil {
		t.Fatalf("different predictions: %s", err.Error())
	}
}

func InnerTestXGBLinRawAgaricus(t *testing.T, nThreads int) {
	testPath := filepath.Join("testdata", "agaricus_test.libsvm")
	modelPath := filepath.Join("testdata", "xgblin_agaricus.model")
	truePath := filepath.Join("testdata", "xgblin_agaricus_true_raw_predictions.txt")
	// loading test data
	csr, err := mat.CSRMatFromLibsvmFile(testPath, 0, true)
	if err != nil {
		t.Fatal(err)
	}

	// loading model with transformation function (binary classification in the case)
	model, err := XGBLinearFromFile(modelPath, false)
	if err != nil {
		t.Fatal(err)
	}

	// loading true predictions as DenseMat
	truePredictions, err := mat.DenseMatFromCsvFile(truePath, 0, false, ",", 0.0)
	if err != nil {
		t.Fatal(err)
	}

	// do predictions
	predictions := make([]float64, csr.Rows())
	model.PredictCSR(csr.RowHeaders, csr.ColIndexes, csr.Values, predictions, 0, nThreads)
	// compare results
	if err := util.AlmostEqualFloat64Slices(truePredictions.Values, predictions, 1e-5); err != nil {
		t.Fatalf("different predictions: %s", err.Error())
	}
}

func TestLGMulticlass(t *testing.T) {
	InnerTestLGMulticlass(t, false, 1)
	InnerTestLGMulticlass(t, false, 2)
	InnerTestLGMulticlass(t, false, 3)
	InnerTestLGMulticlass(t, false, 4)
	InnerTestLGMulticlass(t, true, 1)
	InnerTestLGMulticlass(t, true, 2)
	InnerTestLGMulticlass(t, true, 3)
	InnerTestLGMulticlass(t, true, 4)
}

func InnerTestLGMulticlass(t *testing.T, loadTransformation bool, nThreads int) {
	// loading test data
	testPath := filepath.Join("testdata", "multiclass_test.tsv")
	modelPath := filepath.Join("testdata", "lgmulticlass.model")
	truePath := filepath.Join("testdata", "lgmulticlass_true_raw_predictions.txt")
	if loadTransformation {
		truePath = filepath.Join("testdata", "lgmulticlass_true_predictions.txt")
	}
	skipTestIfFileNotExist(t, testPath, modelPath, truePath)
	dense, err := mat.DenseMatFromCsvFile(testPath, 0, true, "\t", 0.0)
	if err != nil {
		t.Fatal(err)
	}

	// loading model
	model, err := LGEnsembleFromFile(modelPath, loadTransformation)
	if err != nil {
		t.Fatal(err)
	}
	if model.NEstimators() != 10 {
		t.Fatalf("expected 10 trees (got %d)", model.NEstimators())
	}
	if model.NOutputGroups() != 5 {
		t.Fatalf("expected 5 classes (got %d)", model.NOutputGroups())
	}

	// loading true predictions as DenseMat
	truePredictions, err := mat.DenseMatFromCsvFile(truePath, 0, false, "\t", 0.0)
	if err != nil {
		t.Fatal(err)
	}

	// do predictions
	predictions := make([]float64, dense.Rows*model.NOutputGroups())
	model.PredictDense(dense.Values, dense.Rows, dense.Cols, predictions, 0, nThreads)
	// compare results
	const tolerance = 1e-7
	if err := util.AlmostEqualFloat64Slices(truePredictions.Values, predictions, tolerance); err != nil {
		t.Errorf("different predictions: %s", err.Error())
	}

	// check Predict
	singleIdx := 200
	fvals := dense.Values[singleIdx*dense.Cols : (singleIdx+1)*dense.Cols]
	predictions = make([]float64, model.NOutputGroups())
	err = model.Predict(fvals, 0, predictions)
	if err != nil {
		t.Errorf("error while call model.Predict: %s", err.Error())
	}
	if err := util.AlmostEqualFloat64Slices(truePredictions.Values[singleIdx*model.NOutputGroups():(singleIdx+1)*model.NOutputGroups()], predictions, tolerance); err != nil {
		t.Errorf("different Predict prediction: %s", err.Error())
	}
}

func TestXGDermatology(t *testing.T) {
	InnerTestXGDermatology(t, 1)
	InnerTestXGDermatology(t, 2)
	InnerTestXGDermatology(t, 3)
	InnerTestXGDermatology(t, 4)
}

func InnerTestXGDermatology(t *testing.T, nThreads int) {
	// loading test data
	testPath := filepath.Join("testdata", "dermatology_test.libsvm")
	modelPath := filepath.Join("testdata", "xgdermatology.model")
	truePath := filepath.Join("testdata", "xgdermatology_true_predictions.txt")
	skipTestIfFileNotExist(t, testPath, modelPath, truePath)
	csr, err := mat.CSRMatFromLibsvmFile(testPath, 0, true)
	if err != nil {
		t.Fatal(err)
	}

	// loading model
	model, err := XGEnsembleFromFile(modelPath, false)
	if err != nil {
		t.Fatal(err)
	}

	// loading true predictions as DenseMat
	truePredictions, err := mat.DenseMatFromCsvFile(truePath, 0, false, "\t", 0.0)
	if err != nil {
		t.Fatal(err)
	}

	// do predictions
	predictions := make([]float64, csr.Rows()*model.NOutputGroups())
	model.PredictCSR(csr.RowHeaders, csr.ColIndexes, csr.Values, predictions, 0, nThreads)
	// compare results
	const tolerance = 1e-6
	if err := util.AlmostEqualFloat64Slices(truePredictions.Values, predictions, tolerance); err != nil {
		t.Errorf("different predictions: %s", err.Error())
	}
}

func TestSKGradientBoostingClassifier(t *testing.T) {
	// loading test data
	testPath := filepath.Join("testdata", "sk_gradient_boosting_classifier_test.libsvm")
	modelPath := filepath.Join("testdata", "sk_gradient_boosting_classifier.model")
	truePath := filepath.Join("testdata", "sk_gradient_boosting_classifier_true_predictions.txt")
	csr, err := mat.CSRMatFromLibsvmFile(testPath, 0, true)
	if err != nil {
		t.Fatal(err)
	}

	// loading model
	model, err := SKEnsembleFromFile(modelPath, false)
	if err != nil {
		t.Fatal(err)
	}

	// loading true predictions as DenseMat
	truePredictions, err := mat.DenseMatFromCsvFile(truePath, 0, false, "\t", 0.0)
	if err != nil {
		t.Fatal(err)
	}

	// do predictions
	predictions := make([]float64, csr.Rows()*model.NOutputGroups())
	model.PredictCSR(csr.RowHeaders, csr.ColIndexes, csr.Values, predictions, 0, 1)
	// compare results
	const tolerance = 1e-6
	if err := util.AlmostEqualFloat64Slices(truePredictions.Values, predictions, tolerance); err != nil {
		t.Errorf("different predictions: %s", err.Error())
	}
}

func TestSKIoLoadMatchesLegacy(t *testing.T) {
	testPath := filepath.Join("testdata", "sk_gradient_boosting_classifier_test.libsvm")
	modelPath := filepath.Join("testdata", "sk_gradient_boosting_classifier.model")
	csr, err := mat.CSRMatFromLibsvmFile(testPath, 0, true)
	if err != nil {
		t.Fatal(err)
	}
	legacy, err := SKEnsembleFromFile(modelPath, false)
	if err != nil {
		t.Fatal(err)
	}
	fresh, err := io.LoadFromFile(modelPath, &io.LoadOptions{AutoTransform: false, Backend: io.BackendNative})
	if err != nil {
		t.Fatal(err)
	}
	defer fresh.Close()
	want := make([]float64, csr.Rows()*legacy.NOutputGroups())
	legacy.PredictCSR(csr.RowHeaders, csr.ColIndexes, csr.Values, want, 0, 1)
	got := make([]float64, csr.Rows()*fresh.NOutputGroups())
	if err := fresh.PredictCSR(csr.RowHeaders, csr.ColIndexes, csr.Values, got, 0, 1); err != nil {
		t.Fatal(err)
	}
	if err := util.AlmostEqualFloat64Slices(want, got, 1e-6); err != nil {
		t.Errorf("io ForestIR vs legacy SK: %s", err.Error())
	}
}

func TestSKIris(t *testing.T) {
	testPath := filepath.Join("testdata", "iris_test.libsvm")
	modelPath := filepath.Join("testdata", "sk_iris.model")
	truePath := filepath.Join("testdata", "sk_iris_true_predictions.txt")
	skipTestIfFileNotExist(t, testPath, truePath, modelPath)

	// loading test data
	csr, err := mat.CSRMatFromLibsvmFile(testPath, 0, true)
	if err != nil {
		t.Fatal(err)
	}

	// loading model
	model, err := SKEnsembleFromFile(modelPath, false)
	if err != nil {
		t.Fatal(err)
	}

	// loading true predictions as DenseMat
	truePredictions, err := mat.DenseMatFromCsvFile(truePath, 0, false, "\t", 0.0)
	if err != nil {
		t.Fatal(err)
	}

	// do predictions
	predictions := make([]float64, csr.Rows()*model.NOutputGroups())
	model.PredictCSR(csr.RowHeaders, csr.ColIndexes, csr.Values, predictions, 0, 1)
	// compare results
	const tolerance = 1e-6
	// compare results. Count number of mismatched values beacase of floating point
	// comparisons problems: fval <= thresholds.
	// I think this is because float32 format in sklearn X matrix
	count, err := util.NumMismatchedFloat64Slices(truePredictions.Values, predictions, tolerance)
	if err != nil {
		t.Error(err)
	}

	if count > 2 {
		t.Errorf("mismatched more than %d predictions", count)
	}
}

func TestLGRandomForestIris(t *testing.T) {
	testPath := filepath.Join("testdata", "iris_test.libsvm")
	modelPath := filepath.Join("testdata", "lg_rf_iris.model")
	truePath := filepath.Join("testdata", "lg_rf_iris_true_predictions.txt")
	skipTestIfFileNotExist(t, testPath, truePath, modelPath)

	// loading test data
	csr, err := mat.CSRMatFromLibsvmFile(testPath, 0, true)
	if err != nil {
		t.Fatal(err)
	}

	// loading model
	model, err := LGEnsembleFromFile(modelPath, true)
	if err != nil {
		t.Fatal(err)
	}
	if model.Transformation().Name() != "raw" {
		t.Errorf("TransformName should be \"raw\" (got: %s)", model.Transformation().Name())
	}
	if model.Transformation().Type() != transformation.Raw {
		t.Errorf("TransformType should be Raw (got: %d)", model.Transformation().Type())
	}

	// loading true predictions as DenseMat
	truePredictions, err := mat.DenseMatFromCsvFile(truePath, 0, false, "\t", 0.0)
	if err != nil {
		t.Fatal(err)
	}
	// do predictions
	predictions := make([]float64, csr.Rows()*model.NOutputGroups())
	model.PredictCSR(csr.RowHeaders, csr.ColIndexes, csr.Values, predictions, 0, 1)
	// compare results
	const tolerance = 1e-6
	if err := util.AlmostEqualFloat64Slices(truePredictions.Values, predictions, tolerance); err != nil {
		t.Errorf("different predictions: %s", err.Error())
	}
}

func TestXGDARTAgaricus(t *testing.T) {
	// loading test data
	path := filepath.Join("testdata", "agaricus_test.libsvm")
	reader, err := os.Open(path)
	if err != nil {
		t.Skipf("Skipping due to absence of %s", path)
	}
	bufReader := bufio.NewReader(reader)
	csr, err := mat.CSRMatFromLibsvm(bufReader, 0, true)
	if err != nil {
		t.Fatal(err)
	}

	// loading model
	path = filepath.Join("testdata", "xg_dart_agaricus.model")
	model, err := XGEnsembleFromFile(path, false)
	if err != nil {
		t.Fatal(err)
	}

	// loading true predictions as DenseMat
	path = filepath.Join("testdata", "xg_dart_agaricus_true_predictions.txt")
	reader, err = os.Open(path)
	if err != nil {
		t.Skipf("Skipping due to absence of %s", path)
	}
	bufReader = bufio.NewReader(reader)
	truePredictions, err := mat.DenseMatFromCsv(bufReader, 0, false, ",", 0.0)
	if err != nil {
		t.Fatal(err)
	}

	// do predictions
	predictions := make([]float64, csr.Rows())
	model.PredictCSR(csr.RowHeaders, csr.ColIndexes, csr.Values, predictions, 10, 1)
	// compare results
	if err := util.AlmostEqualFloat64Slices(truePredictions.Values, predictions, 1e-5); err != nil {
		t.Fatalf("different predictions: %s", err.Error())
	}
}

func TestLGDARTBreastCancer(t *testing.T) {
	testPath := filepath.Join("testdata", "breast_cancer_test.tsv")
	modelPath := filepath.Join("testdata", "lg_dart_breast_cancer.model")
	truePath := filepath.Join("testdata", "lg_dart_breast_cancer_true_predictions.txt")
	skipTestIfFileNotExist(t, testPath, truePath, modelPath)

	// loading test data
	test, err := mat.DenseMatFromCsvFile(testPath, 0, false, "\t", 0.0)
	if err != nil {
		t.Fatal(err)
	}

	// loading model
	model, err := LGEnsembleFromFile(modelPath, true)
	if err != nil {
		t.Fatal(err)
	}

	// loading true predictions as DenseMat
	truePredictions, err := mat.DenseMatFromCsvFile(truePath, 0, false, "\t", 0.0)
	if err != nil {
		t.Fatal(err)
	}

	// do predictions
	predictions := make([]float64, test.Rows*model.NOutputGroups())
	err = model.PredictDense(test.Values, test.Rows, test.Cols, predictions, 0, 1)
	if err != nil {
		t.Fatal(err)
	}

	// compare results
	const tolerance = 1e-6
	if err := util.AlmostEqualFloat64Slices(truePredictions.Values, predictions, tolerance); err != nil {
		t.Errorf("different predictions: %s", err.Error())
	}
}

// test on categorical variables in LightGBM
func TestLGKDDCup99(t *testing.T) {
	testPath := filepath.Join("testdata", "kddcup99_test.tsv")
	modelPath := filepath.Join("testdata", "lg_kddcup99.model")
	truePath := filepath.Join("testdata", "lg_kddcup99_true_predictions.txt")
	skipTestIfFileNotExist(t, testPath, truePath, modelPath)

	// loading test data
	test, err := mat.DenseMatFromCsvFile(testPath, 0, false, "\t", 0.0)
	if err != nil {
		t.Fatal(err)
	}

	// loading model
	model, err := LGEnsembleFromFile(modelPath, false)
	if err != nil {
		t.Fatal(err)
	}

	// loading true predictions as DenseMat
	truePredictions, err := mat.DenseMatFromCsvFile(truePath, 0, false, "\t", 0.0)
	if err != nil {
		t.Fatal(err)
	}

	// do predictions
	predictions := make([]float64, test.Rows*model.NOutputGroups())
	err = model.PredictDense(test.Values, test.Rows, test.Cols, predictions, 0, 1)
	if err != nil {
		t.Fatal(err)
	}

	// compare results
	const tolerance = 1e-6
	if err := util.AlmostEqualFloat64Slices(truePredictions.Values, predictions, tolerance); err != nil {
		t.Errorf("different predictions: %s", err.Error())
	}
}

func BenchmarkLGKDDCup99_dense_1thread(b *testing.B) {
	InnerBenchmarkLGKDDCup99(b, 1)
}

func BenchmarkLGKDDCup99_dense_4thread(b *testing.B) {
	InnerBenchmarkLGKDDCup99(b, 4)
}

func InnerBenchmarkLGKDDCup99(b *testing.B, nThreads int) {
	// loading test data
	testPath := filepath.Join("testdata", "kddcup99_test_for_bench.tsv")
	modelPath := filepath.Join("testdata", "lg_kddcup99_for_bench.model")
	skipBenchmarkIfFileNotExist(b, testPath, modelPath)
	test, err := mat.DenseMatFromCsvFile(testPath, 0, false, "\t", 0.0)
	if err != nil {
		b.Fatal(err)
	}
	model, err := LGEnsembleFromFile(modelPath, false)
	if err != nil {
		b.Fatal(err)
	}

	// do benchmark
	b.ResetTimer()
	predictions := make([]float64, test.Rows*model.NOutputGroups())
	for i := 0; i < b.N; i++ {
		model.PredictDense(test.Values, test.Rows, test.Cols, predictions, 0, nThreads)
	}
}

func TestLGJsonBreastCancer(t *testing.T) {
	testPath := filepath.Join("testdata", "breast_cancer_test.tsv")
	modelPath := filepath.Join("testdata", "lg_dart_breast_cancer.json")
	truePath := filepath.Join("testdata", "lg_dart_breast_cancer_true_predictions.txt")
	skipTestIfFileNotExist(t, testPath, truePath, modelPath)

	// loading test data
	test, err := mat.DenseMatFromCsvFile(testPath, 0, false, "\t", 0.0)
	if err != nil {
		t.Fatal(err)
	}

	// loading model
	modelFile, err := os.Open(modelPath)
	if err != nil {
		t.Fatal(err)
	}
	defer modelFile.Close()
	model, err := LGEnsembleFromJSON(modelFile, false)
	if err != nil {
		t.Fatal(err)
	}

	// loading true predictions as DenseMat
	truePredictions, err := mat.DenseMatFromCsvFile(truePath, 0, false, "\t", 0.0)
	if err != nil {
		t.Fatal(err)
	}

	// do predictions
	predictions := make([]float64, test.Rows*model.NOutputGroups())
	err = model.PredictDense(test.Values, test.Rows, test.Cols, predictions, 0, 1)
	if err != nil {
		t.Fatal(err)
	}
	util.SigmoidFloat64SliceInplace(predictions)

	// compare results
	const tolerance = 1e-6
	if err := util.AlmostEqualFloat64Slices(truePredictions.Values, predictions, tolerance); err != nil {
		t.Errorf("different predictions: %s", err.Error())
	}
}

func TestGenlinFMTPPoisson(t *testing.T) {
	testPath := filepath.Join("testdata", "genlin_fmtp_poisson_Frequency_features.tsv")
	modelPath := filepath.Join("testdata", "genlin_fmtp_poisson_Frequency.model")
	truePath := filepath.Join("testdata", "genlin_fmtp_poisson_Frequency_true_predictions.txt")

	skipTestIfFileNotExist(t, testPath, truePath, modelPath)

	// loading test data
	test, err := mat.DenseMatFromCsvFile(testPath, 0, false, "\t", 0.0)
	if err != nil {
		t.Fatal(err)
	}

	model, err := LGEnsembleFromFile(modelPath, true)
	if err != nil {
		t.Fatal(err)
	}

	// loading true predictions as DenseMat
	truePredictions, err := mat.DenseMatFromCsvFile(truePath, 0, false, "\t", 0.0)
	if err != nil {
		t.Fatal(err)
	}

	// do predictions
	predictions := make([]float64, test.Rows*model.NOutputGroups())
	err = model.PredictDense(test.Values, test.Rows, test.Cols, predictions, 0, 1)
	if err != nil {
		t.Fatal(err)
	}

	// compare results
	const tolerance = 1e-6
	if err := util.AlmostEqualFloat64Slices(truePredictions.Values, predictions, tolerance); err != nil {
		t.Errorf("different predictions: %s", err.Error())
	}
}

func TestGenlinFMTPGamma(t *testing.T) {
	testPath := filepath.Join("testdata", "genlin_fmtp_gamma_AvgClaimAmount_features.tsv")
	modelPath := filepath.Join("testdata", "genlin_fmtp_gamma_AvgClaimAmount.model")
	truePath := filepath.Join("testdata", "genlin_fmtp_gamma_AvgClaimAmount_true_predictions.txt")

	skipTestIfFileNotExist(t, testPath, truePath, modelPath)

	// loading test data
	test, err := mat.DenseMatFromCsvFile(testPath, 0, false, "\t", 0.0)
	if err != nil {
		t.Fatal(err)
	}

	model, err := LGEnsembleFromFile(modelPath, true)
	if err != nil {
		t.Fatal(err)
	}

	// loading true predictions as DenseMat
	truePredictions, err := mat.DenseMatFromCsvFile(truePath, 0, false, "\t", 0.0)
	if err != nil {
		t.Fatal(err)
	}

	// do predictions
	predictions := make([]float64, test.Rows*model.NOutputGroups())
	err = model.PredictDense(test.Values, test.Rows, test.Cols, predictions, 0, 1)
	if err != nil {
		t.Fatal(err)
	}

	// compare results
	const tolerance = 1e-6
	if err := util.AlmostEqualFloat64Slices(truePredictions.Values, predictions, tolerance); err != nil {
		t.Errorf("different predictions: %s", err.Error())
	}
}

func TestGenlinFMTPTweedie(t *testing.T) {
	testPath := filepath.Join("testdata", "genlin_fmtp_tweedie_PurePremium_features.tsv")
	modelPath := filepath.Join("testdata", "genlin_fmtp_tweedie_PurePremium.model")
	truePath := filepath.Join("testdata", "genlin_fmtp_tweedie_PurePremium_true_predictions.txt")

	skipTestIfFileNotExist(t, testPath, truePath, modelPath)

	// loading test data
	test, err := mat.DenseMatFromCsvFile(testPath, 0, false, "\t", 0.0)
	if err != nil {
		t.Fatal(err)
	}

	model, err := LGEnsembleFromFile(modelPath, true)
	if err != nil {
		t.Fatal(err)
	}

	// loading true predictions as DenseMat
	truePredictions, err := mat.DenseMatFromCsvFile(truePath, 0, false, "\t", 0.0)
	if err != nil {
		t.Fatal(err)
	}

	// do predictions
	predictions := make([]float64, test.Rows*model.NOutputGroups())
	err = model.PredictDense(test.Values, test.Rows, test.Cols, predictions, 0, 1)
	if err != nil {
		t.Fatal(err)
	}

	// compare results
	const tolerance = 1e-6
	if err := util.AlmostEqualFloat64Slices(truePredictions.Values, predictions, tolerance); err != nil {
		t.Errorf("different predictions: %s", err.Error())
	}
}
