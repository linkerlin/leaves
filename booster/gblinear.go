package booster

import (
	"github.com/linkerlin/leaves/data"
	"github.com/linkerlin/leaves/linear"
)

// GBLinearConfig 线性 booster 超参。
type GBLinearConfig struct {
	LearningRate float64
	Lambda       float64
}

// GBLinear 梯度提升线性模型。
type GBLinear struct {
	linear *linear.LinearIR
	cfg    GBLinearConfig
}

// NewGBLinear 创建空 gblinear booster。
func NewGBLinear(numFeatures, numOutputs int, baseScore float64, cfg GBLinearConfig) *GBLinear {
	if numOutputs <= 0 {
		numOutputs = 1
	}
	if cfg.LearningRate <= 0 {
		cfg.LearningRate = 0.5
	}
	w := make([]float64, numFeatures*numOutputs+numOutputs)
	return &GBLinear{
		cfg: cfg,
		linear: &linear.LinearIR{
			NumFeatures:     numFeatures,
			NumOutputGroups: numOutputs,
			BaseScore:       baseScore,
			Weights:         w,
			Name:            "leaves.gblinear",
		},
	}
}

func (b *GBLinear) Linear() *linear.LinearIR { return b.linear }

func (b *GBLinear) setLearningRate(lr float64) { b.cfg.LearningRate = lr }

func (b *GBLinear) NumOutputGroups() int { return b.linear.NumOutputGroups }

// Boost 一轮坐标下降式权重更新。
func (b *GBLinear) Boost(dm data.Matrix, grad, hess []float64) {
	lin := b.linear
	g := lin.NumOutputGroups
	nf := lin.NumFeatures
	n := dm.NumRow()
	row := make([]float64, nf)
	lr := b.cfg.LearningRate
	lambda := b.cfg.Lambda

	for f := 0; f < nf; f++ {
		for k := 0; k < g; k++ {
			var sumG, sumH float64
			for i := 0; i < n; i++ {
				_ = dm.Row(i, row)
				gi, hi := gradHessAt(grad, hess, i, k, g)
				x := row[f]
				sumG += gi * x
				sumH += hi * x * x
			}
			idx := g*f + k
			lin.Weights[idx] += -sumG / (sumH + lambda) * lr
		}
	}
	for k := 0; k < g; k++ {
		var sumG, sumH float64
		for i := 0; i < n; i++ {
			gi, hi := gradHessAt(grad, hess, i, k, g)
			sumG += gi
			sumH += hi
		}
		idx := nf*g + k
		lin.Weights[idx] += -sumG / (sumH + lambda) * lr
	}
}

// PredictMargins 批量 raw margin；多输出时 out 为 [n*groups] 行优先。
func (b *GBLinear) PredictMargins(dm data.Matrix, out []float64) {
	lin := b.linear
	g := lin.NumOutputGroups
	nf := lin.NumFeatures
	n := dm.NumRow()
	row := make([]float64, nf)
	biasOff := nf * g

	for i := 0; i < n; i++ {
		_ = dm.Row(i, row)
		base := i * g
		for k := 0; k < g; k++ {
			sum := lin.BaseScore
			if biasOff+k < len(lin.Weights) {
				sum += lin.Weights[biasOff+k]
			}
			for f := 0; f < nf && f < len(row); f++ {
				idx := g*f + k
				if idx < len(lin.Weights) {
					sum += row[f] * lin.Weights[idx]
				}
			}
			if base+k < len(out) {
				out[base+k] = sum
			}
		}
	}
}
