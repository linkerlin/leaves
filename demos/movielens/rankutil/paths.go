// Package rankutil 提供 MovieLens demo 共用路径与评估工具。
package rankutil

import (
	"fmt"
	"os"
	"path/filepath"
)

// DataDir 返回 testdata 目录（默认从当前工作目录向上查找仓库根）。
func DataDir() (string, error) {
	if v := os.Getenv("LEAVES_TESTDATA"); v != "" {
		return v, nil
	}
	cwd, err := os.Getwd()
	if err != nil {
		return "", err
	}
	dir := cwd
	for i := 0; i < 8; i++ {
		candidate := filepath.Join(dir, "testdata", "rank_movielens_train.tsv")
		if _, err := os.Stat(candidate); err == nil {
			return filepath.Join(dir, "testdata"), nil
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			break
		}
		dir = parent
	}
	return "", fmt.Errorf("testdata not found (run from repo root or set LEAVES_TESTDATA)")
}

// OutDir 返回 demo 输出目录。
func OutDir() (string, error) {
	data, err := DataDir()
	if err != nil {
		return "", err
	}
	out := filepath.Join(filepath.Dir(data), "demos", "movielens", "out")
	if err := os.MkdirAll(out, 0o755); err != nil {
		return "", err
	}
	return out, nil
}
