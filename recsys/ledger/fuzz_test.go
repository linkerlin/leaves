package ledger_test

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/linkerlin/leaves/v2/recsys/ledger"
)

// FuzzLedgerOpen 信任边界模糊测试：任意字节的账本 JSONL 经 ledger.Open
// （ReadJSONL + 三类事件 Unmarshal + 回链校验），不得 panic。
// CI 默认只跑种子语料；深挖：go test ./recsys/ledger -fuzz FuzzLedgerOpen -fuzztime 60s。
func FuzzLedgerOpen(f *testing.F) {
	f.Add([]byte(`{"kind":"decision","data":{"decision_id":"d1","request_id":"r1","occurred_at":"2026-01-01T00:00:00Z","policy_version":"p1","model_sha256":"` + sha256Hex + `","items":[{"item_id":"i1","rank":1,"reason":"ok"}]}}`))
	f.Add([]byte(`{"kind":"exposure","data":{"exposure_id":"e1","decision_id":"d1","occurred_at":"2026-01-01T00:01:00Z","items":[{"item_id":"i1","rank":1}]}}`))
	f.Add([]byte(`{"kind":"feedback","data":{"feedback_id":"f1","exposure_id":"e1","occurred_at":"2026-01-01T00:02:00Z","event_type":"click"}}`))
	f.Add([]byte(`{"kind":"unknown"}`))
	f.Add([]byte(`{bad json`))
	f.Add([]byte("\n\n"))
	f.Fuzz(func(t *testing.T, data []byte) {
		path := filepath.Join(t.TempDir(), "ledger.jsonl")
		if err := os.WriteFile(path, data, 0o644); err != nil {
			t.Skip()
		}
		l, err := ledger.Open(path)
		if err == nil && l == nil {
			t.Fatal("Open returned nil ledger without error")
		}
	})
}

// sha256Hex：合法格式占位（64 hex）；校验逻辑属 contract 层，此处只需种子能走到回链分支。
const sha256Hex = "0000000000000000000000000000000000000000000000000000000000000000"
