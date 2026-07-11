package tree

import (
	"encoding/json"
	"time"
)

// BenchRecordSchemaVersion 统一 benchmark 记录契约版本。
const BenchRecordSchemaVersion = 1

// BenchRecord 是 CI / 本地 bench 可比较的统一 JSON 行（Phase C）。
// 写入 JSONL 或单文件，便于跨 release 对比 ns_per_op。
type BenchRecord struct {
	SchemaVersion int    `json:"schema_version"`
	Name          string `json:"name"`    // 用例名，如 predict/lg_breast/batch1
	Backend       string `json:"backend"` // native|born_cpu|born_gpu
	BatchSize     int    `json:"batch_size"`
	NsPerOp       int64  `json:"ns_per_op"` // 平均纳秒/次
	Iters         int    `json:"iters,omitempty"`
	Model         string `json:"model,omitempty"`
	// AutoRule：若该记录对应 BackendAuto 选型，填 SelectBackendExplained.Rule。
	AutoRule  string `json:"auto_rule,omitempty"`
	CreatedAt string `json:"created_at"` // RFC3339 UTC
	Note      string `json:"note,omitempty"`
}

// NewBenchRecord 构造一条记录（自动填 schema_version 与时间戳）。
func NewBenchRecord(name string, backend Backend, batchSize int, nsPerOp int64) BenchRecord {
	return BenchRecord{
		SchemaVersion: BenchRecordSchemaVersion,
		Name:          name,
		Backend:       BackendName(backend),
		BatchSize:     batchSize,
		NsPerOp:       nsPerOp,
		CreatedAt:     time.Now().UTC().Format(time.RFC3339),
	}
}

// MarshalJSONL 编码为单行 JSON（末尾无换行由调用方追加）。
func (r BenchRecord) MarshalJSONL() ([]byte, error) {
	return json.Marshal(r)
}
