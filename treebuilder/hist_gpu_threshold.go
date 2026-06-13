package treebuilder

const (
	gpuHistMinSamplesFloor = 32
	gpuHistMinSamplesRoot  = 64
)

// gpuHistMinSamplesForDepth 随树深度略降阈值；浅层保持 64，避免小节点 GPU 固定开销反超 CPU。
func gpuHistMinSamplesForDepth(depth int) int {
	switch {
	case depth <= 4:
		return gpuHistMinSamplesRoot
	case depth == 5:
		return 48
	default:
		return gpuHistMinSamplesFloor
	}
}
