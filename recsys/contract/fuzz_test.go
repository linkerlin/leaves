package contract

import (
	"encoding/json"
	"testing"
)

// FuzzValidateInteractionsJSON 信任边界模糊测试：任意 JSON 解为
// []InteractionEvent 后过 ValidateInteractions，不得 panic；
// 空 event_id 的非空批必须被拒绝（校验不可被 fuzz 输入绕过）。
// CI 默认只跑种子语料；深挖：go test ./recsys/contract -fuzz FuzzValidateInteractionsJSON -fuzztime 60s。
func FuzzValidateInteractionsJSON(f *testing.F) {
	f.Add([]byte(`[{"event_id":"e1","occurred_at":"2026-01-01T00:00:00Z","subject_id":"u1","item_id":"i1","event_type":"rating","value":1,"source":"x"}]`))
	f.Add([]byte(`[]`))
	f.Add([]byte(`[{"event_id":""}]`))
	f.Add([]byte(`[{"event_id":"a@a","occurred_at":"2026-01-01T00:00:00Z","subject_id":"u 1","item_id":"i1","event_type":"click"}]`))
	f.Add([]byte(`not json`))
	f.Add([]byte(`[null,0,"x"]`))
	f.Fuzz(func(t *testing.T, raw []byte) {
		var events []InteractionEvent
		if err := json.Unmarshal(raw, &events); err != nil {
			return
		}
		err := ValidateInteractions(events)
		if len(events) > 0 && events[0].EventID == "" && err == nil {
			t.Fatal("empty event_id must be rejected")
		}
	})
}
