package docs_test

import (
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"
	"testing"
)

// TestTestdataMatrixArtifactsExist 锁定 LIB-12：testdata-matrix.md 中反引号引用的
// 稳定/实验模型与数据文件须真实存在于 testdata/（防文档漂移）。
func TestTestdataMatrixArtifactsExist(t *testing.T) {
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("runtime.Caller")
	}
	docsDir := filepath.Dir(file)
	root := filepath.Join(docsDir, "..")
	matrixPath := filepath.Join(docsDir, "testdata-matrix.md")
	body, err := os.ReadFile(matrixPath)
	if err != nil {
		t.Fatal(err)
	}

	// 提取 `...` 中的路径样片段
	re := regexp.MustCompile("`([^`]+)`")
	matches := re.FindAllStringSubmatch(string(body), -1)

	// 仅检查看起来像 testdata 工件的引用
	extOK := map[string]bool{
		".json": true, ".ubj": true, ".model": true, ".txt": true,
		".tsv": true, ".libsvm": true, ".csv": true,
	}
	skipName := map[string]bool{
		"testdata-matrix.md": true,
		"interop-matrix.md":  true,
		"SampleONNXStump":    true, // 内存生成
	}
	// 通配 / 说明性片段
	skipPrefix := []string{"gen_", "rank_*", "shap_contribs_*", "lg_dart_*", "sk_*.model"}
	skipContains := []string{"*", "Test", "go test", "born_parity", "io/", "train/", "data/", "model/", "quantize/", "treebuilder/", "demos/", "wasm", "examples/"}

	var missing []string
	seen := map[string]bool{}
	for _, m := range matches {
		ref := strings.TrimSpace(m[1])
		if ref == "" || seen[ref] {
			continue
		}
		seen[ref] = true
		if skipName[ref] {
			continue
		}
		base := filepath.Base(ref)
		if skipName[base] {
			continue
		}
		// 去掉路径前缀 testdata/
		name := strings.TrimPrefix(ref, "testdata/")
		name = strings.TrimPrefix(name, "testdata\\")
		if strings.Contains(name, "/") || strings.Contains(name, "\\") {
			// 带包路径的测试名等
			skip := false
			for _, s := range skipContains {
				if strings.Contains(ref, s) {
					skip = true
					break
				}
			}
			if skip {
				continue
			}
		}
		ext := strings.ToLower(filepath.Ext(name))
		if !extOK[ext] {
			continue
		}
		// 纯扩展名（文档里写 `.model` / `.txt` 说明性引用）
		if name == ext || strings.TrimPrefix(name, ".") == strings.TrimPrefix(ext, ".") {
			continue
		}
		if len(name) <= len(ext) {
			continue
		}
		// 通配符
		if strings.Contains(name, "*") {
			continue
		}
		for _, p := range skipPrefix {
			if strings.HasPrefix(name, strings.TrimSuffix(p, "*")) && strings.HasSuffix(p, "*") {
				// handled by * check above mostly
				_ = p
			}
		}
		// 仅文件名：落在 testdata/
		path := filepath.Join(root, "testdata", name)
		if _, err := os.Stat(path); err != nil {
			// 也可能文档写了相对路径含子目录
			path2 := filepath.Join(root, name)
			if _, err2 := os.Stat(path2); err2 != nil {
				missing = append(missing, ref)
			}
		}
	}
	if len(missing) > 0 {
		t.Fatalf("testdata-matrix.md references missing files (%d):\n  - %s\n\nUpdate docs or add fixtures under testdata/",
			len(missing), strings.Join(missing, "\n  - "))
	}
	if len(seen) < 10 {
		t.Fatalf("parsed too few refs (%d); regex broken?", len(seen))
	}
}
