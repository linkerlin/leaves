package treebuilder

type gpuHistF32Row struct {
	feat    int
	histG   []float32
	histH   []float32
	numBins int
}

type gpuGainPick struct {
	splitIdx int
	gain     float64
	ok       bool
}
