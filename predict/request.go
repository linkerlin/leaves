package predict

// OutputKind 预测输出类型，对齐 XGBoost PredictionType。
type OutputKind int

const (
	// OutputValue 变换后的最终输出。
	OutputValue OutputKind = iota
	// OutputMargin raw score（累加树/线性输出 + base_score，未做 sigmoid/softmax）。
	OutputMargin
	// OutputLeaf 叶节点索引。
	OutputLeaf
	// OutputContribution Tree SHAP 贡献值（末列 bias，对齐 XGBoost pred_contribs）。
	OutputContribution
	// OutputApproxContribution Saabas 近似贡献值。
	OutputApproxContribution
	// OutputInteraction SHAP 交互值（含 bias 维）。
	OutputInteraction
)

// IterationRange boosting 层切片范围（对齐 XGBoost iteration_begin/end）。
type IterationRange struct {
	Begin int // 含
	End   int // 不含；0 表示到末尾
}

// AllIterations 返回使用全部 boosting 层的范围。
func AllIterations() IterationRange {
	return IterationRange{Begin: 0, End: 0}
}

// Matrix 训练/预测用矩阵抽象（Phase 0.5 先支持稠密与 CSR 切片视图）。
type Matrix interface {
	Kind() MatrixKind
}

// MatrixKind 矩阵存储格式。
type MatrixKind int

const (
	MatrixDense MatrixKind = iota
	MatrixCSR
)

// DenseMatrix 行主序稠密矩阵视图。
type DenseMatrix struct {
	Values []float64
	Rows   int
	Cols   int
}

func (d DenseMatrix) Kind() MatrixKind { return MatrixDense }

// CSRMatrix CSR 稀疏矩阵视图。
type CSRMatrix struct {
	Indptr []int
	Cols   []int
	Values []float64
}

func (c CSRMatrix) Kind() MatrixKind { return MatrixCSR }

// Request 统一预测请求（v1.0 推荐 API）。
type Request struct {
	Matrix      Matrix
	Output      OutputKind
	IterRange   IterationRange
	NEstimators int // 兼容旧 API；0 表示全部
}
