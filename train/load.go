package train

import (
	"fmt"
	"strings"

	"github.com/dmitryikh/leaves/data"
	leavesio "github.com/dmitryikh/leaves/io"
)

// LoadData 从文件加载训练矩阵（自动嗅探 CSV/TSV/libsvm/排序 TSV）。
func LoadData(path string, opts data.FileLoadOptions) (data.Matrix, error) {
	return data.FromFile(path, opts)
}

// LoadDataAuto 等价于 LoadData(path, data.DefaultFileLoadOptions())。
func LoadDataAuto(path string) (data.Matrix, error) {
	return data.FromFileAuto(path)
}

// InferObjectiveFromModel 从 leaves.json / XGBoost JSON·UBJ 读取 objective 名。
func InferObjectiveFromModel(path string) (string, error) {
	format, err := leavesio.DetectFormat(path)
	if err != nil {
		return "", err
	}
	switch format {
	case leavesio.FormatXGBoostJSON:
		r, err := leavesio.ParseXGBoostJSONFile(path)
		if err != nil {
			return "", err
		}
		return r.Objective, nil
	case leavesio.FormatXGBoostUBJSON:
		r, err := leavesio.ParseXGBoostUBJSONFile(path)
		if err != nil {
			return "", err
		}
		return r.Objective, nil
	case leavesio.FormatLeavesJSON:
		r, err := leavesio.LoadLeavesJSONFile(path)
		if err != nil {
			return "", err
		}
		return r.Objective, nil
	default:
		return "", fmt.Errorf("train: cannot infer objective from format %v", format)
	}
}

// NewLearnerFromFile 嗅探数据文件并训练（cfg.Objective 为空时不推断，须显式设置）。
func NewLearnerFromFile(dataPath string, cfg Config, dataOpts data.FileLoadOptions) (*Learner, error) {
	dm, err := LoadData(dataPath, dataOpts)
	if err != nil {
		return nil, fmt.Errorf("train: load data: %w", err)
	}
	learner, err := NewLearner(cfg)
	if err != nil {
		return nil, err
	}
	if err := learner.Fit(dm); err != nil {
		return nil, err
	}
	return learner, nil
}

// NewLearnerFromModelAndData 从已有模型 JSON 推断 objective（若 cfg.Objective 为空）并训练。
func NewLearnerFromModelAndData(modelPath, dataPath string, cfg Config, dataOpts data.FileLoadOptions) (*Learner, error) {
	if strings.TrimSpace(cfg.Objective) == "" {
		obj, err := InferObjectiveFromModel(modelPath)
		if err != nil {
			return nil, fmt.Errorf("train: %w", err)
		}
		cfg.Objective = obj
	}
	_ = modelPath
	return NewLearnerFromFile(dataPath, cfg, dataOpts)
}
