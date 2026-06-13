// Package predict 定义预测运行时接口与请求语义。
// Engine 为树模型与线性模型的统一推理接口。
package predict

// Engine 是可插拔推理后端（树模型、线性模型、GoMLX 等）。
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
