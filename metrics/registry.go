package metrics

import (
	"fmt"
	"strconv"
	"strings"
)

// Options 解析指标时的上下文（多类 / 排序）。
type Options struct {
	NumClass int
	Groups   []int
	NDCGK    int // NDCG@k / MAP@k；0 = 全量
}

// Resolve 按 XGBoost 风格名称解析 Metric（仅走注册表 + 名称归一化）。
// ndcg@K / map@K 会解析 K 后查 "ndcg" / "map" 工厂。
func Resolve(name string, opt Options) (Metric, error) {
	key := NormalizeName(name)
	if key == "" {
		return nil, fmt.Errorf("metrics: empty name")
	}
	// 前缀 @K：写入 Options 后回落到基名注册项。
	if base, k, ok := splitAtK(key); ok {
		opt.NDCGK = k
		key = base
	}
	if f, ok := extraMetrics[key]; ok {
		return f(opt)
	}
	return nil, fmt.Errorf("metrics: unsupported %q (register with metrics.Register)", name)
}

// splitAtK 解析 "ndcg@10" / "map@5" → (ndcg|map, 10, true)。
func splitAtK(key string) (base string, k int, ok bool) {
	i := strings.LastIndex(key, "@")
	if i <= 0 || i == len(key)-1 {
		return "", 0, false
	}
	base = key[:i]
	if base != "ndcg" && base != "map" {
		return "", 0, false
	}
	n, err := strconv.Atoi(key[i+1:])
	if err != nil || n <= 0 {
		return "", 0, false
	}
	return base, n, true
}

// NormalizeName 归一化 XGBoost eval_metric 名称（小写、去空白、常见别名）。
func NormalizeName(name string) string {
	s := strings.TrimSpace(strings.ToLower(name))
	s = strings.ReplaceAll(s, "-", "_")
	switch s {
	case "binary:logistic", "binary_logistic":
		return "logloss"
	case "multi:softmax", "multi:softprob", "multi_softmax", "multi_softprob":
		return "mlogloss"
	case "reg:squarederror", "reg_squarederror":
		return "rmse"
	}
	return s
}

// XGBoostNameTable 返回 leaves 指标与 XGBoost 名称对照（文档 / 训练默认）。
func XGBoostNameTable() map[string]string {
	return map[string]string{
		"rmse":     "reg:squarederror 默认",
		"mae":      "reg:absoluteerror（leaves 用 margin/原值）",
		"mape":     "reg:mape",
		"rmsle":    "reg:rmsle",
		"logloss":  "binary:logistic",
		"error":    "binary:logistic（阈值 0.5）",
		"auc":      "binary:logistic 默认 eval",
		"mlogloss": "multi:softprob",
		"merror":   "multi:softmax",
		"ndcg":     "rank:ndcg",
		"map":      "rank:map",
	}
}

// Evaluate 便捷包装：有 groups 且指标支持时使用 EvaluatePerGroup。
func Evaluate(m Metric, yTrue, yPred []float64, groups []int) (float64, error) {
	if len(groups) > 0 {
		return m.EvaluatePerGroup(yTrue, yPred, groups)
	}
	return m.Evaluate(yTrue, yPred)
}

// RegisteredNames 返回已注册指标名（归一化后）。
func RegisteredNames() []string {
	out := make([]string, 0, len(extraMetrics))
	for k := range extraMetrics {
		out = append(out, k)
	}
	return out
}
