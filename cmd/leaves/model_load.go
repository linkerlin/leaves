package main

import (
	"fmt"
	"path/filepath"
	"strings"

	leavesio "github.com/linkerlin/leaves/v2/io"
	"github.com/linkerlin/leaves/v2/model"
	"github.com/linkerlin/leaves/v2/quantize"
	"github.com/linkerlin/leaves/v2/tree"
)

// loadEnsemble 加载模型：.quant.json 走量化侧车（base+overlay），其余走标准 io。
// 量化引擎输出 raw margin（transform=nil），由调用方按 objective 自行变换。
func loadEnsemble(modelPath string) (*model.Ensemble, error) {
	if strings.HasSuffix(modelPath, ".quant.json") {
		ov, err := quantize.LoadOverlayFile(modelPath)
		if err != nil {
			return nil, fmt.Errorf("load overlay: %w", err)
		}
		base := ov.Base
		if base == "" {
			base = "model.leaves.json"
		}
		basePath := base
		if !filepath.IsAbs(basePath) {
			basePath = filepath.Join(filepath.Dir(modelPath), base)
		}
		ir, _, err := leavesio.ParseLeavesJSONFile(basePath)
		if err != nil {
			return nil, fmt.Errorf("load base model: %w", err)
		}
		if ir.Forest == nil {
			return nil, fmt.Errorf("base model has no forest")
		}
		qf, err := quantize.FromOverlay(ir.Forest, ov)
		if err != nil {
			return nil, err
		}
		eng, err := quantize.NewEngine(qf, nil, tree.TransformRaw, 0)
		if err != nil {
			return nil, err
		}
		return model.NewEnsemble(eng), nil
	}
	if fmtKind, derr := leavesio.DetectFormat(modelPath); derr == nil && fmtKind == leavesio.FormatONNX {
		if ir, _, _, err := leavesio.LoadModelIR(modelPath); err == nil && ir != nil {
			return model.NewEnsembleFromIRWithHint(ir, tree.ApplyTransformRaw, tree.TransformRaw, tree.BackendAuto, tree.WorkloadHint{})
		}
	}
	return leavesio.LoadFromFile(modelPath, &leavesio.LoadOptions{LoadTransformation: false})
}

// loadModelCategoryMaps 从模型文件（或 .quant.json 的 base 模型）提取持久化的类别编码字典（WP-31）。
func loadModelCategoryMaps(modelPath string) map[string]map[string]float64 {
	if strings.HasSuffix(modelPath, ".quant.json") {
		ov, err := quantize.LoadOverlayFile(modelPath)
		if err != nil {
			return nil
		}
		base := ov.Base
		if base == "" {
			base = "model.leaves.json"
		}
		basePath := base
		if !filepath.IsAbs(basePath) {
			basePath = filepath.Join(filepath.Dir(modelPath), base)
		}
		if ir, _, err := leavesio.ParseLeavesJSONFile(basePath); err == nil && ir != nil {
			return ir.CategoryMaps
		}
		return nil
	}
	if ir, _, _, err := leavesio.LoadModelIR(modelPath); err == nil && ir != nil {
		return ir.CategoryMaps
	}
	return nil
}

// loadModelFeatureNames 从模型文件（或 .quant.json 的 base 模型）提取持久化的特征列名（WP-40）。
func loadModelFeatureNames(modelPath string) []string {
	if ir, _, _, _, err := loadModelIRWithQuant(modelPath); err == nil && ir != nil {
		return ir.FeatureNames
	}
	return nil
}

// loadModelIRWithQuant 统一加载模型 IR，若为 .quant.json 则自动解析其 base 模型并返回量化侧车（WP-46）。
func loadModelIRWithQuant(modelPath string) (*model.ModelIR, string, leavesio.Format, *quantize.Overlay, error) {
	if strings.HasSuffix(modelPath, ".quant.json") {
		ov, err := quantize.LoadOverlayFile(modelPath)
		if err != nil {
			return nil, "", leavesio.FormatUnknown, nil, fmt.Errorf("load overlay: %w", err)
		}
		base := ov.Base
		if base == "" {
			base = "model.leaves.json"
		}
		basePath := base
		if !filepath.IsAbs(basePath) {
			basePath = filepath.Join(filepath.Dir(modelPath), base)
		}
		ir, obj, fmtKind, err := leavesio.LoadModelIR(basePath)
		if err != nil {
			return nil, "", leavesio.FormatUnknown, nil, fmt.Errorf("load base model %s: %w", basePath, err)
		}
		return ir, obj, fmtKind, &ov, nil
	}
	ir, obj, fmtKind, err := leavesio.LoadModelIR(modelPath)
	return ir, obj, fmtKind, nil, err
}
