package rankutil

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

// RowMeta ranking 行旁车：movie_id / title（由 gen_rank_movielens.py 写出）。
type RowMeta struct {
	QID     int     `json:"qid"`
	UserID  int     `json:"user_id"`
	Row     int     `json:"row"`
	MovieID int     `json:"movie_id"`
	Title   string  `json:"title"`
	Label   float64 `json:"label"`
}

// LoadTestMeta 加载测试集旁车；文件缺失时返回 (nil, nil)。
func LoadTestMeta(dataDir string) ([]RowMeta, error) {
	return loadMeta(filepath.Join(dataDir, "rank_movielens_test_meta.jsonl"))
}

func loadMeta(path string) ([]RowMeta, error) {
	f, err := os.Open(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	defer f.Close()
	var out []RowMeta
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for sc.Scan() {
		line := sc.Bytes()
		if len(line) == 0 {
			continue
		}
		var m RowMeta
		if err := json.Unmarshal(line, &m); err != nil {
			return nil, fmt.Errorf("meta %s: %w", path, err)
		}
		out = append(out, m)
	}
	return out, sc.Err()
}

// MetaIndex 按 (qid, row) 查元数据。
func MetaIndex(meta []RowMeta) map[[2]int]RowMeta {
	m := make(map[[2]int]RowMeta, len(meta))
	for _, r := range meta {
		m[[2]int{r.QID, r.Row}] = r
	}
	return m
}
