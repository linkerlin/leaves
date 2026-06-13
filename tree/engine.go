package tree

// Backend 表示推理后端类型。
type Backend int

const (
	// BackendNative 纯 Go 原生实现（正确性 golden），零外部 ML 依赖。
	BackendNative Backend = iota
	// BackendBornCPU Born CPU 后端（SIMD 张量加速）。
	BackendBornCPU
	// BackendBornGPU Born WebGPU 后端（Windows DX12 等，零 CGO）。
	BackendBornGPU
	// BackendAuto 由 SelectBackend 根据模型能力与 workload 自动选择。
	BackendAuto Backend = 99
)

// Engine 是树推理的可插拔后端接口。
// 实现：NativeEngine（golden）、BornEngine（Born 张量加速）。
type Engine interface {
	PredictDense(
		vals []float64, nrows int, ncols int,
		predictions []float64,
		nEstimators int,
	) error

	PredictCSR(
		indptr []int, cols []int, vals []float64,
		predictions []float64,
		nEstimators int,
	) error

	PredictSingle(fvals []float64, nEstimators int) float64

	Predict(fvals []float64, nEstimators int, predictions []float64) error

	PredictLeafIndicesDense(
		vals []float64, nrows int, ncols int,
		predictions []float64,
	) error

	PredictLeafIndicesCSR(
		indptr []int, cols []int, vals []float64,
		predictions []float64,
	) error

	NOutputGroups() int
	NRawOutputGroups() int
	NFeatures() int
	NEstimators() int
	NLeaves() []int
	Name() string
	Close() error
}

// Factory 引擎构造函数类型。
type Factory func(forest *ForestIR, transform TransformFn) (Engine, error)

// TransformFn 对原始预测值应用变换（如 sigmoid、softmax）。
type TransformFn func(rawPredictions []float64, outputPredictions []float64, startIndex int) error

// TransformType 变换类型枚举。
type TransformType int

const (
	TransformRaw TransformType = 0
	TransformLogistic TransformType = 1
	TransformSoftmax TransformType = 2
	TransformLeafIndex TransformType = 3
	TransformExponential TransformType = 4
)
