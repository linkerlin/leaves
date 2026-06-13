// Package model 提供 ModelIR 等统一模型中间表示。
package model

import (
	"github.com/dmitryikh/leaves/linear"
	"github.com/dmitryikh/leaves/tree"
)

// ModelKind 模型种类。
type ModelKind int

const (
	// KindGBTree 梯度提升树（XGBoost gbtree、LightGBM gbdt 等）。
	KindGBTree ModelKind = iota
	// KindDART DART 加权树模型。
	KindDART
	// KindGBLinear XGBoost 线性 booster。
	KindGBLinear
	// KindSklearnGBDT scikit-learn 梯度提升。
	KindSklearnGBDT
)

// ModelIR 顶层模型中间表示，统一树模型与线性模型。
type ModelIR struct {
	Kind             ModelKind
	NumFeatures      int
	NRawOutputGroups int
	NOutputGroups    int
	Name             string
	FeatureNames     []string
	FeatureTypes     []string // XGBoost："float" / "c"

	// 树模型（KindGBTree / KindDART / KindSklearnGBDT）
	Forest *tree.ForestIR

	// 线性模型（KindGBLinear）
	Linear *linear.LinearIR
}

// IsTree 是否为树模型。
func (m *ModelIR) IsTree() bool {
	return m.Forest != nil
}

// IsLinear 是否为线性模型。
func (m *ModelIR) IsLinear() bool {
	return m.Linear != nil
}

// IsNumericOnly 是否仅含数值分裂（GoMLX 后端选型用）。
func (m *ModelIR) IsNumericOnly() bool {
	if m.Linear != nil {
		return true
	}
	if m.Forest == nil {
		return true
	}
	for i := range m.Forest.Trees {
		t := &m.Forest.Trees[i]
		for j := range t.IsCategorical {
			if t.IsCategorical[j] {
				return false
			}
		}
	}
	return true
}
