package model

import (
	"github.com/dmitryikh/leaves/explain"
	"github.com/dmitryikh/leaves/tree"
)

// Explainer 模型可解释性 API。
type Explainer struct {
	forest *tree.ForestIR
}

// Explain 返回可解释性接口（需树模型）。
func (e *Ensemble) Explain() *Explainer {
	return &Explainer{forest: e.Forest()}
}

// TreeSHAP 计算 Tree SHAP 值（margin 空间）。
func (x *Explainer) TreeSHAP(features [][]float64) ([][]float64, error) {
	if x == nil || x.forest == nil {
		return nil, nil
	}
	return explain.NewTreeExplainer(x.forest).ShapleyValues(features)
}

// InteractionSHAP 计算 Tree SHAP 交互值（margin 空间）。
func (x *Explainer) InteractionSHAP(features [][]float64) ([][][]float64, error) {
	if x == nil || x.forest == nil {
		return nil, nil
	}
	return explain.NewTreeExplainer(x.forest).InteractionValues(features)
}

// TreeSHAPMulticlass 多类 Tree SHAP，返回 [sample][feature][class]。
func (x *Explainer) TreeSHAPMulticlass(features [][]float64) ([][][]float64, error) {
	if x == nil || x.forest == nil {
		return nil, nil
	}
	return explain.NewTreeExplainer(x.forest).ShapleyValuesMulticlass(features)
}

// InteractionSHAPMulticlass 多类交互 SHAP。
func (x *Explainer) InteractionSHAPMulticlass(features [][]float64) ([][][][]float64, error) {
	if x == nil || x.forest == nil {
		return nil, nil
	}
	return explain.NewTreeExplainer(x.forest).InteractionValuesMulticlass(features)
}

// ExpectedValues 每个类别的 margin 基线。
func (x *Explainer) ExpectedValues() []float64 {
	if x == nil || x.forest == nil {
		return nil
	}
	return explain.NewTreeExplainer(x.forest).ExpectedValues()
}

// Saabas 计算 Saabas 近似贡献（margin 空间）。
func (x *Explainer) Saabas(features [][]float64) ([][]float64, error) {
	if x == nil || x.forest == nil {
		return nil, nil
	}
	return explain.NewTreeExplainer(x.forest).ApproximateContributions(features)
}

// ExpectedValue margin 基线。
func (x *Explainer) ExpectedValue() float64 {
	if x == nil || x.forest == nil {
		return 0
	}
	return explain.NewTreeExplainer(x.forest).ExpectedValue()
}

// Importance 特征重要性。
func (x *Explainer) Importance(kind explain.ImportanceType, names []string) *explain.FeatureImportance {
	if x == nil || x.forest == nil {
		return nil
	}
	return explain.ComputeImportance(x.forest, kind, names)
}

// DumpText 文本 dump。
func (x *Explainer) DumpText(names []string) string {
	if x == nil || x.forest == nil {
		return ""
	}
	return explain.DumpText(x.forest, names)
}

// DumpJSON JSON dump。
func (x *Explainer) DumpJSON(names []string) ([]byte, error) {
	if x == nil || x.forest == nil {
		return nil, nil
	}
	return explain.DumpJSON(x.forest, names)
}

// DumpDOT Graphviz DOT dump。
func (x *Explainer) DumpDOT(names []string) string {
	if x == nil || x.forest == nil {
		return ""
	}
	return explain.DumpDOT(x.forest, names)
}
