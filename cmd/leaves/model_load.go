package main

import (
	"fmt"
	"path/filepath"
	"strings"

	leavesio "github.com/linkerlin/leaves/io"
	"github.com/linkerlin/leaves/model"
	"github.com/linkerlin/leaves/quantize"
	"github.com/linkerlin/leaves/tree"
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
	return leavesio.LoadFromFile(modelPath, &leavesio.LoadOptions{LoadTransformation: false})
}
