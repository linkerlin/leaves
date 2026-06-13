package objective

// ParseRankPairMethod 解析 lambdarank_pair_method（full|topk|mean）。
func ParseRankPairMethod(s string) RankPairMethod {
	switch s {
	case "topk", "top_k":
		return RankPairTopK
	case "mean":
		return RankPairMean
	default:
		return RankPairFull
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
