package treebuilder

// leafBudget 检查 lossguide MaxLeaves 限制。
func leafBudgetExceeded(cfg Config, leaves *int) bool {
	return leaves != nil && cfg.MaxLeaves > 0 && *leaves >= cfg.MaxLeaves
}

func splitLeafBudget(cfg Config, leaves *int) {
	if leaves != nil && cfg.MaxLeaves > 0 {
		*leaves++
	}
}

func intPtr1() *int { n := 1; return &n }

func splitBudget(cfg Config, leaves *int) *int {
	splitLeafBudget(cfg, leaves)
	return leaves
}
