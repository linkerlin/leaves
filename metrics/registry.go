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
	NDCGK    int // NDCG@k；0 = 全量
}

// Resolve 按 XGBoost 风格名称解析内置 Metric。
// 支持别名：logloss↔logloss、binary:logistic→error（仅名称归一化，不含 objective 映射）。
func Resolve(name string, opt Options) (Metric, error) {
	key := NormalizeName(name)
	if key == "" {
		return nil, fmt.Errorf("metrics: empty name")
	}
	switch key {
	case "rmse":
		return RMSE{}, nil
	case "mae":
		return MAE{}, nil
	case "mape":
		return MAPE{}, nil
	case "rmsle":
		return RMSLE{}, nil
	case "logloss":
		return LogLoss{}, nil
	case "error", "binary_error":
		return Error{}, nil
	case "auc", "aucpr":
		return AUC{}, nil
	case "mlogloss":
		if opt.NumClass < 2 {
			return nil, fmt.Errorf("metrics: mlogloss needs num_class >= 2")
		}
		return MLogLoss{NumClass: opt.NumClass}, nil
	case "merror":
		if opt.NumClass < 2 {
			return nil, fmt.Errorf("metrics: merror needs num_class >= 2")
		}
		return MError{NumClass: opt.NumClass}, nil
	case "ndcg":
		return NDCG{RankingMetric: RankingMetric{Groups: opt.Groups, K: opt.NDCGK}}, nil
	case "map":
		return MAP{RankingMetric: RankingMetric{Groups: opt.Groups, K: opt.NDCGK}}, nil
	default:
		if strings.HasPrefix(key, "ndcg@") {
			k, err := strconv.Atoi(key[5:])
			if err != nil || k <= 0 {
				return nil, fmt.Errorf("metrics: invalid %q", name)
			}
			return NDCG{RankingMetric: RankingMetric{Groups: opt.Groups, K: k}}, nil
		}
		return nil, fmt.Errorf("metrics: unsupported %q", name)
	}
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
