package model

import "github.com/linkerlin/leaves/tree"

func capsFromIR(ir *ModelIR) tree.ModelCaps {
	if ir == nil {
		return tree.ModelCaps{}
	}
	return tree.ModelCapsFromForest(ir.Forest, ir.IsLinear(), ir.IsNumericOnly())
}

// ResolveBackend 解析 ModelIR 的后端选择（含 BackendAuto）。
func ResolveBackend(ir *ModelIR, requested tree.Backend, hint tree.WorkloadHint) tree.Backend {
	return tree.ResolveBackend(requested, capsFromIR(ir), hint)
}

// SelectBackend 根据 ModelIR 与 workload 自动选择后端。
func SelectBackend(ir *ModelIR, hint tree.WorkloadHint) tree.Backend {
	return tree.SelectBackend(capsFromIR(ir), hint)
}
