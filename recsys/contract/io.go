package contract

import (
	"bufio"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
)

// HashFile 计算文件内容 sha256（hex）。
func HashFile(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", fmt.Errorf("contract: open %s: %w", path, err)
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", fmt.Errorf("contract: hash %s: %w", path, err)
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

// FeatureSchemaHash 特征契约指纹：列名顺序与默认值纳入 hash。
// 变更列名/顺序/默认值都会改变指纹，从而阻断误用旧模型评估新特征。
func FeatureSchemaHash(names []string, defaults []float64) string {
	if len(defaults) > 0 && len(defaults) != len(names) {
		defaults = nil
	}
	var b strings.Builder
	for i, n := range names {
		b.WriteString(n)
		if defaults != nil {
			fmt.Fprintf(&b, "\x00%v", defaults[i])
		}
		b.WriteByte(';')
	}
	sum := sha256.Sum256([]byte(b.String()))
	return hex.EncodeToString(sum[:])
}

// WriteJSONL 逐行写 JSON 数组元素（时间字段须已为 UTC）。
func WriteJSONL[T any](path string, rows []T) error {
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("contract: create %s: %w", path, err)
	}
	defer f.Close()
	w := bufio.NewWriter(f)
	enc := json.NewEncoder(w)
	for i := range rows {
		if err := enc.Encode(rows[i]); err != nil {
			return fmt.Errorf("contract: encode %s row %d: %w", path, i, err)
		}
	}
	if err := w.Flush(); err != nil {
		return fmt.Errorf("contract: flush %s: %w", path, err)
	}
	return nil
}

// VerifyFiles 重新计算快照输入文件 hash 并与记录比对（RC-02：任一不匹配即失败）。
// path 以 root 为基准解析，root 为空表示绝对/当前目录路径。
func VerifyFiles(s *DatasetSnapshot, root string) error {
	for i, f := range s.InputFiles {
		p := f.Path
		if root != "" && !filepath.IsAbs(p) {
			p = filepath.Join(root, p)
		}
		got, err := HashFile(p)
		if err != nil {
			return fmt.Errorf("contract: snapshot %s input[%d]: %w", s.SnapshotID, i, err)
		}
		if got != f.SHA256 {
			return fmt.Errorf("contract: snapshot %s input[%d] %s hash mismatch: recorded %s got %s",
				s.SnapshotID, i, f.Path, f.SHA256, got)
		}
	}
	return nil
}

// ReadJSONL 逐行读 JSONL。
func ReadJSONL[T any](path string) ([]T, error) {
	var out []T
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("contract: open %s: %w", path, err)
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 64*1024), 16*1024*1024)
	line := 0
	for sc.Scan() {
		line++
		s := strings.TrimSpace(sc.Text())
		if s == "" {
			continue
		}
		var row T
		if err := json.Unmarshal([]byte(s), &row); err != nil {
			return nil, fmt.Errorf("contract: decode %s line %d: %w", path, line, err)
		}
		out = append(out, row)
	}
	if err := sc.Err(); err != nil {
		return nil, fmt.Errorf("contract: scan %s: %w", path, err)
	}
	return out, nil
}
