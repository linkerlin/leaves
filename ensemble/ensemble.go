// Package ensemble 提供 GBRT 集成模型的具体实现（gbtree, dart, gblinear）。
//
// Phase 0: 继续使用根包中的现有 lgEnsemble/xgEnsemble/xgLinear 实现。
// Phase 1: 重构为以 tree.ForestIR + tree.Engine 为基础的全新实现。
package ensemble

// Booster 所有 booster 的统一表示。
// Phase 1 中将成为构建 tree.ForestIR 的入口。
type Booster struct {
	Name     string
	ForestIR interface{} // tree.ForestIR (Phase 1)
}
