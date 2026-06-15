package model

import "github.com/linkerlin/leaves/tree"

// forestSource 可返回底层 ForestIR 的引擎。
type forestSource interface {
	Forest() *tree.ForestIR
}

// Forest 返回树模型的 ForestIR；线性模型返回 nil。
func (e *Ensemble) Forest() *tree.ForestIR {
	if fs, ok := e.engine.(forestSource); ok {
		return fs.Forest()
	}
	return nil
}
