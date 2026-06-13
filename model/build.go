package model

import (
	"fmt"

	"github.com/dmitryikh/leaves/linear"
	"github.com/dmitryikh/leaves/predict"
	"github.com/dmitryikh/leaves/tree"
)

// NewEngine 从 ModelIR 创建推理引擎。
func NewEngine(ir *ModelIR, transform tree.TransformFn, outputType tree.TransformType, backend tree.Backend) (predict.Engine, error) {
	return NewEngineWithHint(ir, transform, outputType, backend, tree.DefaultWorkloadHint())
}

// NewEngineWithHint 创建引擎，BackendAuto 时按 hint 自动选择。
func NewEngineWithHint(
	ir *ModelIR,
	transform tree.TransformFn,
	outputType tree.TransformType,
	backend tree.Backend,
	hint tree.WorkloadHint,
) (predict.Engine, error) {
	if ir == nil {
		return nil, fmt.Errorf("nil ModelIR")
	}
	backend = ResolveBackend(ir, backend, hint)

	switch {
	case ir.Linear != nil:
		return linear.NewNativeEngine(ir.Linear, transform, outputType, ir.NOutputGroups), nil
	case ir.Forest != nil:
		switch backend {
		case tree.BackendBornCPU, tree.BackendBornGPU:
			cfg := &tree.BornConfig{UseGPU: backend == tree.BackendBornGPU}
			return tree.NewBornEngine(ir.Forest, transform, outputType, ir.NOutputGroups, cfg)
		default:
			return tree.NewNativeEngine(ir.Forest, transform, outputType, ir.NOutputGroups), nil
		}
	default:
		return nil, fmt.Errorf("ModelIR has no booster data")
	}
}

// NewEnsembleFromIR 从 ModelIR 创建 model.Ensemble。
func NewEnsembleFromIR(ir *ModelIR, transform tree.TransformFn, outputType tree.TransformType, backend tree.Backend) (*Ensemble, error) {
	return NewEnsembleFromIRWithHint(ir, transform, outputType, backend, tree.DefaultWorkloadHint())
}

// NewEnsembleFromIRWithHint 从 ModelIR 创建 Ensemble，支持 BackendAuto。
func NewEnsembleFromIRWithHint(
	ir *ModelIR,
	transform tree.TransformFn,
	outputType tree.TransformType,
	backend tree.Backend,
	hint tree.WorkloadHint,
) (*Ensemble, error) {
	engine, err := NewEngineWithHint(ir, transform, outputType, backend, hint)
	if err != nil {
		return nil, err
	}
	return NewEnsemble(engine), nil
}
