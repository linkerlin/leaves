package io_test

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/linkerlin/leaves/v2/io"
	// 根包 init 注册遗留 loader/builder；io_test 外部包导入不构成环。
	_ "github.com/linkerlin/leaves/v2"
)

// FuzzLoadFromFileBytes 信任边界模糊测试：任意字节经 DetectFormat + 遗留
// loader + engine builder，必须返回错误或模型，不得 panic / 不得「无错返回 nil」。
// CI 默认只跑种子语料；深挖：go test ./io -fuzz FuzzLoadFromFileBytes -fuzztime 60s。
func FuzzLoadFromFileBytes(f *testing.F) {
	for _, seed := range []string{
		"../testdata/breast_cancer_xgb_baseline.json",
		"../testdata/_v.json",
	} {
		if b, err := os.ReadFile(seed); err == nil {
			f.Add(b)
		}
	}
	f.Add([]byte(`{}`))
	f.Add([]byte(`{"format":"leaves-json"}`))
	f.Add([]byte("\x00\x00bin"))
	f.Add([]byte("0 qid:1 1:0.5\n"))
	f.Add([]byte("User,f0,label\n1,2,0\n"))
	f.Fuzz(func(t *testing.T, data []byte) {
		path := filepath.Join(t.TempDir(), "model") // 无扩展名：走内容嗅探路径
		if err := os.WriteFile(path, data, 0o644); err != nil {
			t.Skip()
		}
		m, err := io.LoadFromFile(path, nil)
		if err == nil && m == nil {
			t.Fatal("LoadFromFile returned nil model without error")
		}
	})
}
