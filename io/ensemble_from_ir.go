package io

import (
	"github.com/linkerlin/leaves/v2/model"
)

// ensembleFromIR 从 ModelIR 构建推荐路径 Ensemble（不依赖根包 init）。
func ensembleFromIR(ir *model.ModelIR, objective string, opts *LoadOptions) (*model.Ensemble, error) {
	if ir == nil {
		return nil, ErrFormatNotImplemented("nil model ir")
	}
	if opts == nil {
		opts = DefaultLoadOptions()
	}
	loadTransform := ResolveLoadTransformation(opts, objective)
	backend := opts.Backend
	hint := opts.Workload
	outType, transform := ObjectiveToTransform(objective, loadTransform)
	ir.NOutputGroups = NOutputGroupsForTransform(ir.NRawOutputGroups, outType)
	return model.NewEnsembleFromIRWithHint(ir, transform, outType, backend, hint)
}
