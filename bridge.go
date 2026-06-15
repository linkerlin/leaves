package leaves

import (
	"github.com/linkerlin/leaves/linear"
	"github.com/linkerlin/leaves/model"
	"github.com/linkerlin/leaves/predict"
	"github.com/linkerlin/leaves/transformation"
	"github.com/linkerlin/leaves/tree"
)

// ---- 类型别名：向后兼容 ----

type TransformType = transformation.TransformType
type Transform = transformation.Transform

const (
	TransformRaw         = transformation.Raw
	TransformLogistic    = transformation.Logistic
	TransformSoftmax     = transformation.Softmax
	TransformLeafIndex   = transformation.LeafIndex
	TransformExponential = transformation.Exponential
)

// ---- IR 转换 ----

func lgTreeToLgNodeData(t *lgTree) []tree.LgNodeData {
	nodes := make([]tree.LgNodeData, len(t.nodes))
	for i, n := range t.nodes {
		nodes[i] = tree.LgNodeData{
			Threshold: n.Threshold,
			Left:      n.Left,
			Right:     n.Right,
			Feature:   n.Feature,
			Flags:     n.Flags,
		}
	}
	return nodes
}

func lgTreeToTreeIR(lt *lgTree) *tree.TreeIR {
	nodes := lgTreeToLgNodeData(lt)
	return tree.BuildTreeIR(nodes, lt.leafValues, lt.catBoundaries, lt.catThresholds, lt.nCategorical)
}

func lgEnsembleToForestIR(e *lgEnsemble) *tree.ForestIR {
	trees := make([]tree.TreeIR, len(e.Trees))
	for i, lt := range e.Trees {
		t := lgTreeToTreeIR(&lt)
		trees[i] = *t
	}

	wd := make([]float64, len(e.Trees))
	for i := range wd {
		wd[i] = 1.0
	}

	return &tree.ForestIR{
		NumFeatures:     e.NFeatures(),
		NumOutputGroups: e.nRawOutputGroups,
		Trees:           trees,
		BaseScore:       0.0,
		WeightDrop:      wd,
		AverageOutput:   e.averageOutput,
		Name:            e.Name(),
	}
}

func xgEnsembleToForestIR(e *xgEnsemble) *tree.ForestIR {
	trees := make([]tree.TreeIR, len(e.Trees))
	for i, lt := range e.Trees {
		t := lgTreeToTreeIR(&lt)
		trees[i] = *t
	}

	treeInfo, iterationIndptr := xgbIterationMeta(len(e.Trees), e.nRawOutputGroups)

	return &tree.ForestIR{
		NumFeatures:       e.NFeatures(),
		NumOutputGroups:   e.nRawOutputGroups,
		Trees:             trees,
		BaseScore:         e.BaseScore,
		WeightDrop:        e.WeightDrop,
		AverageOutput:     false,
		Name:              e.Name(),
		TreeInfo:          treeInfo,
		IterationIndptr:   iterationIndptr,
	}
}

// xgbIterationMeta 从标准 XGBoost 树布局推导 TreeInfo 与 IterationIndptr。
func xgbIterationMeta(numTrees, numOutputGroups int) ([]int, []int) {
	if numOutputGroups <= 0 {
		numOutputGroups = 1
	}
	treeInfo := make([]int, numTrees)
	for i := range treeInfo {
		treeInfo[i] = i % numOutputGroups
	}
	nIter := numTrees / numOutputGroups
	if nIter == 0 {
		return treeInfo, nil
	}
	indptr := make([]int, nIter+1)
	for i := range indptr {
		indptr[i] = i * numOutputGroups
	}
	return treeInfo, indptr
}

func xgLinearToLinearIR(e *xgLinear) *linear.LinearIR {
	weights := make([]float64, len(e.Weights))
	for i, w := range e.Weights {
		weights[i] = float64(w)
	}
	return &linear.LinearIR{
		NumFeatures:     e.NumFeature,
		NumOutputGroups: e.nRawOutputGroups,
		BaseScore:       e.BaseScore,
		Weights:         weights,
		Name:            "xgboost.gblinear",
	}
}

func isDartForest(forest *tree.ForestIR) bool {
	for _, w := range forest.WeightDrop {
		if w != 1.0 {
			return true
		}
	}
	return false
}

// EnsembleToModelIR 将根包 Ensemble 转为统一 ModelIR。
func EnsembleToModelIR(e *Ensemble) *model.ModelIR {
	if e == nil {
		return nil
	}

	base := model.ModelIR{
		NOutputGroups: e.NOutputGroups(),
	}

	switch impl := e.ensembleCore().(type) {
	case *lgEnsemble:
		forest := lgEnsembleToForestIR(impl)
		base.Kind = model.KindGBTree
		if impl.averageOutput {
			base.Kind = model.KindSklearnGBDT // RF 模式近似
		}
		base.NumFeatures = forest.NumFeatures
		base.NRawOutputGroups = forest.NumOutputGroups
		base.Name = forest.Name
		base.Forest = forest
	case *xgEnsemble:
		forest := xgEnsembleToForestIR(impl)
		base.Kind = model.KindGBTree
		if isDartForest(forest) {
			base.Kind = model.KindDART
		}
		base.NumFeatures = forest.NumFeatures
		base.NRawOutputGroups = forest.NumOutputGroups
		base.Name = forest.Name
		base.Forest = forest
	case *xgLinear:
		lin := xgLinearToLinearIR(impl)
		base.Kind = model.KindGBLinear
		base.NumFeatures = lin.NumFeatures
		base.NRawOutputGroups = lin.NumOutputGroups
		base.Name = lin.Name
		base.Linear = lin
	default:
		return nil
	}

	return &base
}

// transformToTreeTransform 将 transformation.Transform 适配为 tree.TransformFn。
func transformToTreeTransform(t transformation.Transform) tree.TransformFn {
	switch t.Type() {
	case transformation.Raw:
		return tree.ApplyTransformRaw
	case transformation.Logistic:
		return tree.ApplyTransformLogistic
	case transformation.Softmax:
		return tree.ApplyTransformSoftmax
	case transformation.Exponential:
		return tree.ApplyTransformExponential
	default:
		return tree.ApplyTransformRaw
	}
}

// EngineOptions 创建引擎时的选项。
type EngineOptions struct {
	Backend  tree.Backend
	Workload tree.WorkloadHint
}

// DefaultEngineOptions 默认 BackendAuto（小 batch 选 Native，大 batch 可 Born）。
func DefaultEngineOptions() *EngineOptions {
	return &EngineOptions{
		Backend:  tree.BackendAuto,
		Workload: tree.DefaultWorkloadHint(),
	}
}

// NewEngineFromEnsemble 从现有 Ensemble 创建 predict.Engine。
func NewEngineFromEnsemble(e *Ensemble) (predict.Engine, error) {
	return NewEngineFromEnsembleWithOptions(e, DefaultEngineOptions())
}

// NewEngineFromEnsembleWithOptions 带后端选项创建引擎。
func NewEngineFromEnsembleWithOptions(e *Ensemble, opts *EngineOptions) (predict.Engine, error) {
	ir := EnsembleToModelIR(e)
	if ir == nil {
		return nil, nil
	}
	if opts == nil {
		opts = DefaultEngineOptions()
	}
	outputType := tree.TransformType(e.transform.Type())
	transformFn := transformToTreeTransform(e.transform)
	return model.NewEngineWithHint(ir, transformFn, outputType, opts.Backend, opts.Workload)
}

// NewModelEnsemble 从根包 Ensemble 创建 model.Ensemble（推荐新 API 入口）。
func NewModelEnsemble(e *Ensemble, opts *EngineOptions) (*model.Ensemble, error) {
	engine, err := NewEngineFromEnsembleWithOptions(e, opts)
	if err != nil {
		return nil, err
	}
	if engine == nil {
		return nil, nil
	}
	return model.NewEnsemble(engine), nil
}
