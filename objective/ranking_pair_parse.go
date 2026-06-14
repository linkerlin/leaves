package objective

// ParseRankPairMethod 解析 lambdarank_pair_method（full|topk|mean）；空串默认 topk（对标 XGBoost）。
func ParseRankPairMethod(s string) RankPairMethod {
	switch s {
	case "full":
		return RankPairFull
	case "topk", "top_k":
		return RankPairTopK
	case "mean":
		return RankPairMean
	default:
		return RankPairTopK
	}
}

// RankPairMethodName 返回配置字符串。
func RankPairMethodName(m RankPairMethod) string {
	switch m {
	case RankPairTopK:
		return "topk"
	case RankPairMean:
		return "mean"
	default:
		return "full"
	}
}
