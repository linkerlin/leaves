package io

// ObjectiveNeedsTransform 判断 objective 是否应在 LoadTransformation/AutoTransform 下启用变换。
func ObjectiveNeedsTransform(objective string) bool {
	switch objective {
	case "binary:logistic", "reg:logistic",
		"multi:softprob", "multi:softmax",
		"reg:gamma", "count:poisson", "reg:tweedie":
		return true
	}
	return false
}

// ResolveLoadTransformation 合并 LoadTransformation 与 AutoTransform。
func ResolveLoadTransformation(opts *LoadOptions, objective string) bool {
	if opts != nil && opts.LoadTransformation {
		return true
	}
	if opts != nil && opts.AutoTransform && ObjectiveNeedsTransform(objective) {
		return true
	}
	return false
}
