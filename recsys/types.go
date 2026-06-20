package recsys

// Interaction 四元交互：User/Item/Tag 为 String，Score 为 float64。
type Interaction struct {
	User  string
	Item  string
	Tag   string
	Score float64
}

// CatalogItem 物品目录行：Item/Tag + 数值特征列。
type CatalogItem struct {
	Item  string
	Tag   string
	Feats []float64
}

// UserQID 用户至 qid 映射。
type UserQID struct {
	User  string
	QID   int
	Split string // train | test
}

// RecallRow 召回候选行。
type RecallRow struct {
	User        string
	Item        string
	Tag         string
	RecallScore float64
	Feats       []float64
}

// ManifestRow 排序 TSV 行映射；Score 为 LTR margin（推理后填入）。
type ManifestRow struct {
	User        string  `json:"User"`
	Item        string  `json:"Item"`
	Tag         string  `json:"Tag"`
	RecallScore float64 `json:"recall_score"`
	Score       float64 `json:"Score,omitempty"`
}

// DealRow 发牌终稿行。
type DealRow struct {
	User  string
	Item  string
	Tag   string
	Score float64
	Rank  int
}

// PrepReport 数据准备统计。
type PrepReport struct {
	Stage       string         `json:"stage"`
	TrainUsers  int            `json:"train_users"`
	TestUsers   int            `json:"test_users"`
	TrainRows   int            `json:"train_rows"`
	TestRows    int            `json:"test_rows"`
	CatalogSize int            `json:"catalog_size"`
	TagVocab    []string       `json:"tag_vocab"`
	Dropped     map[string]int `json:"dropped"`
}

// SmokeConfig 端到端 smoke 参数。
type SmokeConfig struct {
	Seed        int64
	TrainUsers  int
	TestUsers   int
	RecallSize  int
	NumItems    int
	MinEvents   int
	DeckSize    int
	MaxSameTag  int
	TrainRounds int
	NDCGK       int
}

// DefaultSmokeConfig 默认 smoke 参数（100 Item/User）。
func DefaultSmokeConfig() SmokeConfig {
	return SmokeConfig{
		Seed:        42,
		TrainUsers:  18,
		TestUsers:   6,
		RecallSize:  100,
		NumItems:    512,
		MinEvents:   12,
		DeckSize:    10,
		MaxSameTag:  3,
		TrainRounds: 25,
		NDCGK:       10,
	}
}
