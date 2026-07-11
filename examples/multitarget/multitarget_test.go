package main

import "testing"

func TestMultiTargetExampleSmoke(t *testing.T) {
	// main 可执行路径过重；此处只保证包可编译。
	// 完整逻辑见 train.TestMultiTargetOneOutputPerTree。
	if testing.Short() {
		t.Skip("short")
	}
}
