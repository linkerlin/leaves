package docs_test

import (
	"bufio"
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"github.com/linkerlin/leaves/tree"
)

// TestReleaseDocsPresent 锁定 Phase E：发版文档与关键矩阵文件存在。
func TestReleaseDocsPresent(t *testing.T) {
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("runtime.Caller")
	}
	docsDir := filepath.Dir(file)
	root := filepath.Join(docsDir, "..")

	required := []string{
		filepath.Join(docsDir, "release-checklist.md"),
		filepath.Join(docsDir, "release-notes-v2.1.0.md"),
		filepath.Join(docsDir, "release-notes-v2.1.1.md"),
		filepath.Join(docsDir, "versioning.md"),
		filepath.Join(docsDir, "api-surface.md"),
		filepath.Join(docsDir, "interop-matrix.md"),
		filepath.Join(docsDir, "backend-auto.md"),
		filepath.Join(docsDir, "extension-points.md"),
		filepath.Join(docsDir, "testdata-matrix.md"),
		filepath.Join(docsDir, "benchmark-baseline.md"),
		filepath.Join(docsDir, "bench", "sample_benchrecords.jsonl"),
		filepath.Join(docsDir, "bench", "README.md"),
		filepath.Join(root, "NOTES.md"),
		filepath.Join(root, "CHANGELOG.md"),
		filepath.Join(root, "README.md"),
		filepath.Join(root, "README.en.md"),
		filepath.Join(root, "演进计划.md"),
		filepath.Join(root, "演进方案.md"),
		filepath.Join(root, "TODO.md"),
	}
	for _, p := range required {
		st, err := os.Stat(p)
		if err != nil || st.IsDir() {
			t.Errorf("missing required doc: %s (%v)", p, err)
		}
	}
}

// TestREADMELanguageCrossLinks 中英文 README 互链不得指向不存在的文件。
func TestREADMELanguageCrossLinks(t *testing.T) {
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("runtime.Caller")
	}
	root := filepath.Join(filepath.Dir(file), "..")
	zh, err := os.ReadFile(filepath.Join(root, "README.md"))
	if err != nil {
		t.Fatal(err)
	}
	en, err := os.ReadFile(filepath.Join(root, "README.en.md"))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(zh), "README.en.md") {
		t.Error("README.md should link to README.en.md")
	}
	if strings.Contains(string(en), "README.zh.md") {
		t.Error("README.en.md must not link to missing README.zh.md; use README.md")
	}
	if !strings.Contains(string(en), "](README.md)") {
		t.Error("README.en.md should link to Chinese README.md")
	}
}

// TestBenchSampleArtifact 锁定 LIB-02：样例 JSONL 存在且符合 tree.BenchRecord 契约。
func TestBenchSampleArtifact(t *testing.T) {
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("runtime.Caller")
	}
	path := filepath.Join(filepath.Dir(file), "bench", "sample_benchrecords.jsonl")
	f, err := os.Open(path)
	if err != nil {
		t.Fatalf("open sample: %v", err)
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	n := 0
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		var r tree.BenchRecord
		if err := json.Unmarshal([]byte(line), &r); err != nil {
			t.Fatalf("line %d: %v\n%s", n+1, err, line)
		}
		if r.SchemaVersion != tree.BenchRecordSchemaVersion {
			t.Fatalf("line %d schema_version=%d want %d", n+1, r.SchemaVersion, tree.BenchRecordSchemaVersion)
		}
		if r.Name == "" || r.Backend == "" || r.BatchSize <= 0 || r.NsPerOp <= 0 {
			t.Fatalf("line %d incomplete: %+v", n+1, r)
		}
		n++
	}
	if err := sc.Err(); err != nil {
		t.Fatal(err)
	}
	if n < 3 {
		t.Fatalf("expected ≥3 sample records, got %d", n)
	}
}
