package data

// GroupedMatrix 带 query/group 信息的矩阵（排序学习 T5）。
type GroupedMatrix interface {
	Matrix
	Groups() []int
}

// ExternalBatch 外存矩阵单批视图（T5 external memory 草案）。
type ExternalBatch struct {
	Rows   int
	Cols   int
	Labels []float64
	Weights []float64
	// RowAt 将第 i 行写入 buf（长度 ≥ Cols）。
	RowAt func(row int, buf []float64) error
}

// ExternalMemoryMatrix 流式/外存 DMatrix 接口草案（对标 XGBoost QuantileDMatrix）。
// 实现方可按块迭代，训练器在 T5 接入直方图分箱缓存。
type ExternalMemoryMatrix interface {
	Matrix
	// NumBatches 数据划分的批次数；1 表示单块全量。
	NumBatches() int
	// Batch 返回第 b 批（0 ≤ b < NumBatches）。
	Batch(b int) (*ExternalBatch, error)
}
