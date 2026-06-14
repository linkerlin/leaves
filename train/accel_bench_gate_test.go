package train_test

import "testing"

func TestAccelBenchSkippedByDefault(t *testing.T) {
	// 无 LEAVES_BENCH 时由 skipUnlessAccelBench 跳过；此处仅验证 helper 可调用。
	skipUnlessAccelBench(t)
}
