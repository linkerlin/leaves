package predict_test

import (
	"testing"

	"github.com/linkerlin/leaves/predict"
)

func TestAllIterations(t *testing.T) {
	r := predict.AllIterations()
	if r.Begin != 0 || r.End != 0 {
		t.Fatalf("AllIterations = %+v want {0 0}", r)
	}
}

func TestDenseMatrixKind(t *testing.T) {
	d := predict.DenseMatrix{Values: []float64{1, 2, 3}, Rows: 1, Cols: 3}
	if d.Kind() != predict.MatrixDense {
		t.Fatalf("DenseMatrix.Kind() = %v want MatrixDense", d.Kind())
	}
}

func TestCSRMatrixKind(t *testing.T) {
	c := predict.CSRMatrix{Indptr: []int{0, 2}, Cols: []int{0, 1}, Values: []float64{1, 2}}
	if c.Kind() != predict.MatrixCSR {
		t.Fatalf("CSRMatrix.Kind() = %v want MatrixCSR", c.Kind())
	}
}

// TestOutputKindsDistinct locks the OutputKind enum values as distinct.
func TestOutputKindsDistinct(t *testing.T) {
	ks := []predict.OutputKind{
		predict.OutputValue, predict.OutputMargin, predict.OutputLeaf,
		predict.OutputContribution, predict.OutputApproxContribution, predict.OutputInteraction,
	}
	for i, a := range ks {
		for j, b := range ks {
			if i != j && a == b {
				t.Fatalf("OutputKind constants collide at %d/%d = %d", i, j, a)
			}
		}
	}
}

// mockEngine 锁定 Engine 接口契约：接口签名变更时本测试编译失败。
type mockEngine struct{}

func (mockEngine) PredictDense([]float64, int, int, []float64, int) error       { return nil }
func (mockEngine) PredictCSR([]int, []int, []float64, []float64, int) error     { return nil }
func (mockEngine) PredictSingle([]float64, int) float64                         { return 0 }
func (mockEngine) Predict([]float64, int, []float64) error                      { return nil }
func (mockEngine) PredictLeafIndicesDense([]float64, int, int, []float64) error { return nil }
func (mockEngine) PredictLeafIndicesCSR([]int, []int, []float64, []float64) error {
	return nil
}
func (mockEngine) NOutputGroups() int    { return 1 }
func (mockEngine) NRawOutputGroups() int { return 1 }
func (mockEngine) NFeatures() int        { return 1 }
func (mockEngine) NEstimators() int      { return 1 }
func (mockEngine) NLeaves() []int        { return nil }
func (mockEngine) Name() string          { return "mock" }
func (mockEngine) Close() error          { return nil }

func TestEngineInterfaceContract(t *testing.T) {
	var e predict.Engine = mockEngine{}
	if e.Name() != "mock" {
		t.Fatal("mockEngine not wired")
	}
}
