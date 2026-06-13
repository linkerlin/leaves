package treebuilder

// monotoneConstraint 返回特征 f 的单调约束：1 递增，-1 递减，0 无约束。
func monotoneConstraint(cfg Config, feat int) int {
	if feat < 0 || feat >= len(cfg.MonotoneConstraints) {
		return 0
	}
	return cfg.MonotoneConstraints[feat]
}

// splitRespectsMonotone 检查左右子叶权重是否满足单调约束（与 XGBoost 叶值检查一致）。
func splitRespectsMonotone(constraint int, leftVal, rightVal float64) bool {
	if constraint > 0 {
		return leftVal <= rightVal+1e-9
	}
	if constraint < 0 {
		return leftVal >= rightVal-1e-9
	}
	return true
}

func childLeafValues(
	left, right []int,
	grad, hess []float64,
	cfg Config,
) (leftVal, rightVal float64) {
	gl, hl := sumGradHess(left, grad, hess)
	gr, hr := sumGradHess(right, grad, hess)
	lr := cfg.LearningRate
	if lr <= 0 {
		lr = 0.3
	}
	return leafWeightFromSums(gl, hl, cfg.Lambda) * lr,
		leafWeightFromSums(gr, hr, cfg.Lambda) * lr
}

func monotoneAllowsSplit(
	cfg Config,
	feat int,
	left, right []int,
	grad, hess []float64,
) bool {
	c := monotoneConstraint(cfg, feat)
	if c == 0 {
		return true
	}
	lv, rv := childLeafValues(left, right, grad, hess, cfg)
	return splitRespectsMonotone(c, lv, rv)
}
