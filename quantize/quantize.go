// Package quantize 提供树分裂阈值的 int8 仿射量化与相对 Native 的 parity 门禁。
// 叶子值保持 float64；仅数值分裂节点量化，分类节点原样保留。
package quantize

import (
	"fmt"
	"math"

	"github.com/dmitryikh/leaves/predict"
	"github.com/dmitryikh/leaves/tree"
)

// Levels int8 正半轴量化级数（0..Levels）。
const Levels = 127

// Config 量化参数。
type Config struct {
	// Levels 每特征量化级数；0 表示默认 Levels。
	Levels int
}

// QuantizedForest 阈值 int8 量化后的森林（叶子 float 不变）。
type QuantizedForest struct {
	Forest       tree.ForestIR
	QThreshold   [][]int8
	Quantized    [][]bool
	FeatureMin   []float64
	FeatureSpan  []float64
	levels       int
}

// QuantizeForest 对数值分裂阈值做 per-feature int8 仿射量化。
func QuantizeForest(f *tree.ForestIR, cfg Config) (*QuantizedForest, error) {
	if f == nil {
		return nil, fmt.Errorf("quantize: nil forest")
	}
	levels := cfg.Levels
	if levels <= 0 {
		levels = Levels
	}
	nFeat := f.NumFeatures
	if nFeat <= 0 {
		nFeat = 1
	}
	minV := make([]float64, nFeat)
	maxV := make([]float64, nFeat)
	for i := range minV {
		minV[i] = math.MaxFloat64
		maxV[i] = -math.MaxFloat64
	}
	for ti := range f.Trees {
		t := &f.Trees[ti]
		for ni := 0; ni < t.NumNodes; ni++ {
			if isCategoricalNode(t, ni) {
				continue
			}
			feat := int(t.SplitFeature[ni])
			if feat < 0 || feat >= nFeat {
				continue
			}
			th := t.SplitThreshold[ni]
			if th < minV[feat] {
				minV[feat] = th
			}
			if th > maxV[feat] {
				maxV[feat] = th
			}
		}
	}
	span := make([]float64, nFeat)
	for i := range span {
		if maxV[i] >= minV[i] {
			span[i] = maxV[i] - minV[i]
		}
	}

	qf := &QuantizedForest{
		Forest:      *cloneForest(f),
		FeatureMin:  minV,
		FeatureSpan: span,
		levels:      levels,
	}
	qf.QThreshold = make([][]int8, len(f.Trees))
	qf.Quantized = make([][]bool, len(f.Trees))
	for ti := range f.Trees {
		t := &f.Trees[ti]
		qf.QThreshold[ti] = make([]int8, t.NumNodes)
		qf.Quantized[ti] = make([]bool, t.NumNodes)
		for ni := 0; ni < t.NumNodes; ni++ {
			if isCategoricalNode(t, ni) {
				continue
			}
			feat := int(t.SplitFeature[ni])
			if feat < 0 || feat >= nFeat {
				continue
			}
			q := encodeThreshold(t.SplitThreshold[ni], minV[feat], span[feat], levels)
			qf.QThreshold[ti][ni] = q
			qf.Quantized[ti][ni] = true
		}
	}
	return qf, nil
}

func encodeThreshold(th, min, span float64, levels int) int8 {
	if span <= 0 {
		return 0
	}
	q := int(math.Round((th - min) / span * float64(levels)))
	if q < 0 {
		q = 0
	}
	if q > levels {
		q = levels
	}
	return int8(q)
}

func decodeThreshold(q int8, min, span float64, levels int) float64 {
	if levels <= 0 {
		levels = Levels
	}
	return min + float64(q)/float64(levels)*span
}

func isCategoricalNode(t *tree.TreeIR, nodeIdx int) bool {
	return nodeIdx < len(t.IsCategorical) && t.IsCategorical[nodeIdx]
}

func cloneForest(f *tree.ForestIR) *tree.ForestIR {
	if f == nil {
		return nil
	}
	out := *f
	out.Trees = make([]tree.TreeIR, len(f.Trees))
	for i := range f.Trees {
		out.Trees[i] = cloneTree(&f.Trees[i])
	}
	if f.TreeInfo != nil {
		out.TreeInfo = append([]int(nil), f.TreeInfo...)
	}
	if f.IterationIndptr != nil {
		out.IterationIndptr = append([]int(nil), f.IterationIndptr...)
	}
	if f.BaseScores != nil {
		out.BaseScores = append([]float64(nil), f.BaseScores...)
	}
	if f.WeightDrop != nil {
		out.WeightDrop = append([]float64(nil), f.WeightDrop...)
	}
	return &out
}

func cloneTree(t *tree.TreeIR) tree.TreeIR {
	out := *t
	out.SplitFeature = append([]int32(nil), t.SplitFeature...)
	out.SplitThreshold = append([]float64(nil), t.SplitThreshold...)
	out.DefaultLeft = append([]bool(nil), t.DefaultLeft...)
	out.MissingZero = append([]bool(nil), t.MissingZero...)
	out.MissingNan = append([]bool(nil), t.MissingNan...)
	out.LeftChild = append([]int32(nil), t.LeftChild...)
	out.RightChild = append([]int32(nil), t.RightChild...)
	out.LeafValue = append([]float64(nil), t.LeafValue...)
	out.IsCategorical = append([]bool(nil), t.IsCategorical...)
	out.CatOneHot = append([]bool(nil), t.CatOneHot...)
	out.CatSmall = append([]bool(nil), t.CatSmall...)
	out.CatBoundaries = append([]uint32(nil), t.CatBoundaries...)
	out.CatThresholds = append([]uint32(nil), t.CatThresholds...)
	return out
}

// MaxThresholdQuantError 返回数值节点 |原始阈值 - 反量化阈值| 的最大值。
func (qf *QuantizedForest) MaxThresholdQuantError() float64 {
	if qf == nil {
		return 0
	}
	var maxErr float64
	for ti := range qf.Forest.Trees {
		t := &qf.Forest.Trees[ti]
		for ni := 0; ni < t.NumNodes; ni++ {
			if ni >= len(qf.Quantized[ti]) || !qf.Quantized[ti][ni] {
				continue
			}
			feat := int(t.SplitFeature[ni])
			if feat < 0 || feat >= len(qf.FeatureMin) {
				continue
			}
			orig := t.SplitThreshold[ni]
			dq := decodeThreshold(qf.QThreshold[ti][ni], qf.FeatureMin[feat], qf.FeatureSpan[feat], qf.levels)
			err := math.Abs(orig - dq)
			if err > maxErr {
				maxErr = err
			}
		}
	}
	return maxErr
}

// Engine 量化阈值推理引擎（叶子 float；实现 predict.Engine）。
type Engine struct {
	qf           *QuantizedForest
	transform    tree.TransformFn
	outputType   tree.TransformType
	nRawGroups   int
	nOutputGroups int
}

// NewEngine 创建量化推理引擎。
func NewEngine(qf *QuantizedForest, transform tree.TransformFn, outputType tree.TransformType, nOutputGroups int) (*Engine, error) {
	if qf == nil {
		return nil, fmt.Errorf("quantize: nil quantized forest")
	}
	g := qf.Forest.NumOutputGroups
	if g <= 0 {
		g = 1
	}
	if nOutputGroups <= 0 {
		nOutputGroups = g
	}
	return &Engine{
		qf:            qf,
		transform:     transform,
		outputType:    outputType,
		nRawGroups:    g,
		nOutputGroups: nOutputGroups,
	}, nil
}

var _ predict.Engine = (*Engine)(nil)

func (e *Engine) Forest() *tree.ForestIR { return &e.qf.Forest }

func (e *Engine) NOutputGroups() int    { return e.nOutputGroups }
func (e *Engine) NRawOutputGroups() int { return e.nRawGroups }
func (e *Engine) NFeatures() int        { return e.qf.Forest.NumFeatures }
func (e *Engine) NEstimators() int      { return e.qf.Forest.NEstimators() }
func (e *Engine) NLeaves() []int        { return e.qf.Forest.NLeaves() }
func (e *Engine) Name() string          { return e.qf.Forest.Name + ".quantized" }
func (e *Engine) Close() error          { return nil }

func (e *Engine) PredictSingle(fvals []float64, nEstimators int) float64 {
	if e.NOutputGroups() != 1 {
		return 0
	}
	if e.NFeatures() > len(fvals) {
		return 0
	}
	ret := [1]float64{forestMarginsQ(e.qf, fvals, nEstimators)[0]}
	e.applyTransform(ret[:], ret[:], 0)
	return ret[0]
}

func (e *Engine) Predict(fvals []float64, nEstimators int, predictions []float64) error {
	if len(predictions) < e.NOutputGroups() {
		return fmt.Errorf("quantize: predictions too short")
	}
	if e.NFeatures() > len(fvals) {
		return fmt.Errorf("quantize: feature count mismatch")
	}
	m := forestMarginsQ(e.qf, fvals, nEstimators)
	copy(predictions, m)
	e.applyTransform(predictions, predictions, 0)
	return nil
}

func (e *Engine) PredictDense(vals []float64, nrows, ncols int, predictions []float64, nEstimators int) error {
	if len(predictions) < e.NOutputGroups()*nrows {
		return fmt.Errorf("quantize: predictions too short")
	}
	if ncols == 0 || e.NFeatures() > ncols {
		return fmt.Errorf("quantize: column count mismatch")
	}
	g := e.NOutputGroups()
	for i := 0; i < nrows; i++ {
		fvals := vals[i*ncols : (i+1)*ncols]
		m := forestMarginsQ(e.qf, fvals, nEstimators)
		off := i * g
		copy(predictions[off:off+g], m)
	}
	if e.transform != nil && e.outputType != tree.TransformRaw {
		for i := 0; i < nrows; i++ {
			off := i * g
			e.applyTransform(predictions[off:off+g], predictions, off)
		}
	}
	return nil
}

func (e *Engine) PredictCSR(indptr, cols []int, vals []float64, predictions []float64, nEstimators int) error {
	nrows := len(indptr) - 1
	if len(predictions) < e.NOutputGroups()*nrows {
		return fmt.Errorf("quantize: predictions too short")
	}
	fvals := make([]float64, e.NFeatures())
	g := e.NOutputGroups()
	for i := 0; i < nrows; i++ {
		for j := range fvals {
			fvals[j] = math.NaN()
		}
		for j := indptr[i]; j < indptr[i+1]; j++ {
			c := cols[j]
			if c >= 0 && c < len(fvals) {
				fvals[c] = vals[j]
			}
		}
		m := forestMarginsQ(e.qf, fvals, nEstimators)
		off := i * g
		copy(predictions[off:off+g], m)
	}
	if e.transform != nil && e.outputType != tree.TransformRaw {
		for i := 0; i < nrows; i++ {
			off := i * g
			e.applyTransform(predictions[off:off+g], predictions, off)
		}
	}
	return nil
}

func (e *Engine) PredictLeafIndicesDense(vals []float64, nrows, ncols int, predictions []float64) error {
	return fmt.Errorf("quantize: leaf indices not implemented")
}

func (e *Engine) PredictLeafIndicesCSR(indptr, cols []int, vals []float64, predictions []float64) error {
	return fmt.Errorf("quantize: leaf indices not implemented")
}

func (e *Engine) applyTransform(raw, output []float64, start int) {
	if e.transform == nil || e.outputType == tree.TransformRaw {
		return
	}
	e.transform(raw, output, start)
}
