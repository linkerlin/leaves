package leaves

import (
	"math"
	"path/filepath"
	"testing"

	"github.com/linkerlin/leaves/mat"
)

// TestBridgeLightGBM 用真实 LightGBM 模型验证新旧引擎输出一致。
func TestBridgeLightGBM(t *testing.T) {
	modelPath := filepath.Join("testdata", "lg_breast_cancer.txt")
	dataPath := filepath.Join("testdata", "lg_breast_cancer_data.txt")

	// 1. 加载模型（旧 API）
	model, err := LGEnsembleFromFile(modelPath, false)
	if err != nil {
		t.Fatalf("load model: %v", err)
	}

	// 2. 加载数据
	testMat, err := mat.DenseMatFromCsvFile(dataPath, 0, false, " ", 0.0)
	if err != nil {
		t.Fatalf("load data: %v", err)
	}

	// 3. 旧引擎预测
	oldPred := make([]float64, testMat.Rows*model.NOutputGroups())
	err = model.PredictDense(testMat.Values, testMat.Rows, testMat.Cols, oldPred, 0, 0)
	if err != nil {
		t.Fatalf("old predict: %v", err)
	}

	// 4. 创建新引擎
	engine, err := NewEngineFromEnsemble(model)
	if err != nil {
		t.Fatalf("new engine: %v", err)
	}
	defer engine.Close()

	// 5. 新引擎预测
	newPred := make([]float64, testMat.Rows*engine.NOutputGroups())
	err = engine.PredictDense(testMat.Values, testMat.Rows, testMat.Cols, newPred, 0)
	if err != nil {
		t.Fatalf("new predict: %v", err)
	}

	// 6. 逐值对比
	maxDiff := 0.0
	mismatches := 0
	for i := 0; i < len(oldPred); i++ {
		diff := math.Abs(oldPred[i] - newPred[i])
		if diff > maxDiff {
			maxDiff = diff
		}
		if diff > 1e-9 {
			mismatches++
		}
		if diff > 1e-6 {
			t.Errorf("prediction mismatch at %d: old=%f new=%f (diff=%e)",
				i, oldPred[i], newPred[i], diff)
		}
	}
	t.Logf("total predictions: %d, mismatches: %d, max diff: %e",
		len(oldPred), mismatches, maxDiff)
}

// TestBridgeXGBoost 用真实 XGBoost 模型验证新旧引擎输出一致。
func TestBridgeXGBoost(t *testing.T) {
	modelPath := filepath.Join("testdata", "xgagaricus.model")

	model, err := XGEnsembleFromFile(modelPath, false)
	if err != nil {
		t.Fatalf("load xgboost: %v", err)
	}

	engine, err := NewEngineFromEnsemble(model)
	if err != nil {
		t.Fatalf("new engine: %v", err)
	}
	defer engine.Close()

	// 用全零特征值测试
	nFeatures := engine.NFeatures()
	fvals := make([]float64, nFeatures)

	oldPred := model.PredictSingle(fvals, 0)
	newPred := engine.PredictSingle(fvals, 0)

	diff := math.Abs(oldPred - newPred)
	if diff > 1e-9 {
		t.Errorf("xgboost predict mismatch: old=%f new=%f (diff=%e)", oldPred, newPred, diff)
	}

	// 批量测试
	rows := 100
	vals := make([]float64, rows*nFeatures)
	oldPreds := make([]float64, rows*model.NOutputGroups())
	newPreds := make([]float64, rows*engine.NOutputGroups())

	_ = model.PredictDense(vals, rows, nFeatures, oldPreds, 0, 0)
	_ = engine.PredictDense(vals, rows, nFeatures, newPreds, 0)

	maxDiff := 0.0
	for i := 0; i < len(oldPreds); i++ {
		diff := math.Abs(oldPreds[i] - newPreds[i])
		if diff > maxDiff {
			maxDiff = diff
		}
		if diff > 1e-9 {
			t.Errorf("batch mismatch at %d: old=%f new=%f (diff=%e)", i, oldPreds[i], newPreds[i], diff)
		}
	}
	t.Logf("batch %d rows, max diff: %e", rows, maxDiff)
}

// TestBridgeXGBoostCSR 验证 CSR + logistic 变换路径与旧实现一致。
func TestBridgeXGBoostCSR(t *testing.T) {
	testPath := filepath.Join("testdata", "agaricus_test.libsvm")
	modelPath := filepath.Join("testdata", "xgagaricus.model")
	truePath := filepath.Join("testdata", "xgagaricus_true_predictions.txt")

	csr, err := mat.CSRMatFromLibsvmFile(testPath, 0, true)
	if err != nil {
		t.Skipf("testdata not available: %v", err)
	}
	model, err := XGEnsembleFromFile(modelPath, true)
	if err != nil {
		t.Fatal(err)
	}
	engine, err := NewEngineFromEnsemble(model)
	if err != nil {
		t.Fatal(err)
	}
	defer engine.Close()

	oldPred := make([]float64, csr.Rows())
	newPred := make([]float64, csr.Rows())
	_ = model.PredictCSR(csr.RowHeaders, csr.ColIndexes, csr.Values, oldPred, 0, 1)
	_ = engine.PredictCSR(csr.RowHeaders, csr.ColIndexes, csr.Values, newPred, 0)

	for i := range oldPred {
		if math.Abs(oldPred[i]-newPred[i]) > 1e-6 {
			t.Errorf("row %d: legacy=%f engine=%f", i, oldPred[i], newPred[i])
		}
	}

	trueMat, err := mat.DenseMatFromCsvFile(truePath, 0, false, ",", 0.0)
	if err != nil {
		t.Fatalf("load golden: %v", err)
	}
	if len(trueMat.Values) != len(newPred) {
		t.Fatalf("golden size %d != predictions %d", len(trueMat.Values), len(newPred))
	}
	for i := range newPred {
		if math.Abs(newPred[i]-trueMat.Values[i]) > 1e-6 {
			t.Errorf("golden row %d: got=%f want=%f", i, newPred[i], trueMat.Values[i])
		}
	}
}

// TestBridgeAllTrees 确保所有树都被使用。
func TestBridgeAllTrees(t *testing.T) {
	modelPath := filepath.Join("testdata", "lg_breast_cancer.txt")

	model, err := LGEnsembleFromFile(modelPath, false)
	if err != nil {
		t.Fatal(err)
	}

	engine, err := NewEngineFromEnsemble(model)
	if err != nil {
		t.Fatal(err)
	}
	defer engine.Close()

	// 验证元数据一致
	if model.NEstimators() != engine.NEstimators() {
		t.Errorf("NEstimators: old=%d new=%d", model.NEstimators(), engine.NEstimators())
	}
	if model.NFeatures() != engine.NFeatures() {
		t.Errorf("NFeatures: old=%d new=%d", model.NFeatures(), engine.NFeatures())
	}
	if model.NRawOutputGroups() != engine.NRawOutputGroups() {
		t.Errorf("NRawOutputGroups: old=%d new=%d", model.NRawOutputGroups(), engine.NRawOutputGroups())
	}
}
