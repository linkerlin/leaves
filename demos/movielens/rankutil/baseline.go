package rankutil

import (
	"encoding/json"
	"os"
	"path/filepath"
)

// XGBBaseline MovieLens XGBoost 训练基准（由 gen_rank_movielens.py 生成）。
type XGBBaseline struct {
	Objective        string    `json:"objective"`
	NDCGK            int       `json:"ndcg_k"`
	NumRound         int       `json:"num_round"`
	MaxDepth         int       `json:"max_depth"`
	LearningRate     float64   `json:"learning_rate"`
	Lambda           float64   `json:"lambda"`
	Seed             int       `json:"seed"`
	FinalTrainNDCG   float64   `json:"final_train_ndcg"`
	FinalTestNDCG    float64   `json:"final_test_ndcg"`
	InitialTrainNDCG float64   `json:"initial_train_ndcg"`
	InitialTestNDCG  float64   `json:"initial_test_ndcg"`
	TrainNDCG        []float64 `json:"train_ndcg"`
	TestNDCG         []float64 `json:"test_ndcg"`
}

// LoadXGBBaseline 读取 baseline JSON。
func LoadXGBBaseline(path string) (*XGBBaseline, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var bl XGBBaseline
	if err := json.Unmarshal(b, &bl); err != nil {
		return nil, err
	}
	return &bl, nil
}

// BaselinePath 按目标函数返回 baseline 文件路径。
func BaselinePath(dataDir, objective string) string {
	switch objective {
	case "rank:pairwise":
		return filepath.Join(dataDir, "rank_movielens_pairwise_xgb_baseline.json")
	default:
		return filepath.Join(dataDir, "rank_movielens_ndcg_xgb_baseline.json")
	}
}
