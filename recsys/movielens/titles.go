package movielens

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// WriteTitles 写 Item\tTitle 旁车（供发牌展示）。
func WriteTitles(path string, titles map[string]string) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	keys := make([]string, 0, len(titles))
	for k := range titles {
		keys = append(keys, k)
	}
	sort.Slice(keys, func(i, j int) bool {
		// numeric-ish item ids first
		return keys[i] < keys[j]
	})
	var b strings.Builder
	b.WriteString("Item\tTitle\n")
	for _, k := range keys {
		// titles may contain tabs; collapse
		t := strings.ReplaceAll(titles[k], "\t", " ")
		b.WriteString(k)
		b.WriteByte('\t')
		b.WriteString(t)
		b.WriteByte('\n')
	}
	return os.WriteFile(path, []byte(b.String()), 0o644)
}

// LoadTitles 读 Item\tTitle。
func LoadTitles(path string) (map[string]string, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	out := map[string]string{}
	for i, line := range strings.Split(string(raw), "\n") {
		line = strings.TrimSpace(line)
		if line == "" || (i == 0 && strings.HasPrefix(line, "Item\t")) {
			continue
		}
		parts := strings.SplitN(line, "\t", 2)
		if len(parts) < 2 {
			return nil, fmt.Errorf("movielens: bad title line %d", i+1)
		}
		out[parts[0]] = parts[1]
	}
	return out, nil
}
