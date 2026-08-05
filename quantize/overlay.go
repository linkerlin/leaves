package quantize

import (
	"encoding/json"
	"fmt"
	"os"

	"github.com/linkerlin/leaves/v2/tree"
)

// Overlay 是量化参数的可序列化侧车：引用 base 模型的 ForestIR，只存量化叠加层。
// 这样持久化不重复整棵树（base leaves.json 已含完整森林），重建时 base + overlay 即可。
type Overlay struct {
	Base        string    `json:"base,omitempty"` // 相对 base leaves.json（重建时解析）
	Levels      int       `json:"levels"`
	FeatureMin  []float64 `json:"feature_min"`
	FeatureSpan []float64 `json:"feature_span"`
	QThreshold  [][]int8  `json:"q_threshold"`
	Quantized   [][]bool  `json:"quantized"`
}

// Overlay 从已量化的森林导出可持久化侧车。
func (qf *QuantizedForest) Overlay(base string) Overlay {
	return Overlay{
		Base:        base,
		Levels:      qf.levels,
		FeatureMin:  qf.FeatureMin,
		FeatureSpan: qf.FeatureSpan,
		QThreshold:  qf.QThreshold,
		Quantized:   qf.Quantized,
	}
}

// FromOverlay 用 base ForestIR + 量化 overlay 重建 QuantizedForest（重建 inference 用）。
func FromOverlay(f *tree.ForestIR, ov Overlay) (*QuantizedForest, error) {
	if f == nil {
		return nil, fmt.Errorf("quantize: nil base forest")
	}
	levels := ov.Levels
	if levels <= 0 {
		levels = Levels
	}
	if len(ov.QThreshold) != len(f.Trees) {
		return nil, fmt.Errorf("quantize: overlay q_threshold trees %d != base trees %d", len(ov.QThreshold), len(f.Trees))
	}
	for ti, qt := range ov.QThreshold {
		if len(qt) != f.Trees[ti].NumNodes {
			return nil, fmt.Errorf("quantize: overlay tree %d nodes %d != base %d", ti, len(qt), f.Trees[ti].NumNodes)
		}
	}
	return &QuantizedForest{
		Forest:      *f, // 推理只读；浅拷贝足够
		FeatureMin:  ov.FeatureMin,
		FeatureSpan: ov.FeatureSpan,
		QThreshold:  ov.QThreshold,
		Quantized:   ov.Quantized,
		levels:      levels,
	}, nil
}

// SaveOverlayFile 把量化侧车写成 JSON。
func SaveOverlayFile(path, base string, qf *QuantizedForest) error {
	b, err := json.MarshalIndent(qf.Overlay(base), "", "  ")
	if err != nil {
		return err
	}
	b = append(b, '\n')
	return os.WriteFile(path, b, 0o644)
}

// LoadOverlayFile 读取量化侧车 JSON。
func LoadOverlayFile(path string) (Overlay, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return Overlay{}, err
	}
	var ov Overlay
	if err := json.Unmarshal(b, &ov); err != nil {
		return Overlay{}, err
	}
	return ov, nil
}
