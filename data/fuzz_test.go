package data

import (
	"os"
	"path/filepath"
	"testing"
)

// FuzzSniffFileFormatBytes 信任边界模糊测试：任意字节的训练数据文件经
// SniffFileFormat（内容嗅探）与 DetectFileFormat，不得 panic。
// CI 默认只跑种子语料；深挖：go test ./data -fuzz FuzzSniffFileFormatBytes -fuzztime 60s。
func FuzzSniffFileFormatBytes(f *testing.F) {
	f.Add([]byte("a,b,label\n1,2,0\n2,3,1\n"))
	f.Add([]byte("1,2,0\n2,3,1\n"))
	f.Add([]byte("0 qid:1 1:0.5 2:0.2\n1 qid:1 1:0.1\n"))
	f.Add([]byte("0\t1:1.0\t2:0.3\n1\t3:0.2\n"))
	f.Add([]byte(""))
	f.Add([]byte("\xff\xfe\x00garbage"))
	f.Add([]byte("a;b;c\n1;2;3\n"))
	f.Fuzz(func(t *testing.T, data []byte) {
		path := filepath.Join(t.TempDir(), "train")
		if err := os.WriteFile(path, data, 0o644); err != nil {
			t.Skip()
		}
		_, _ = SniffFileFormat(path)
		_ = DetectFileFormat(path)
	})
}
