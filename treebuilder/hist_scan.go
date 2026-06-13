package treebuilder

// scanHistGainsCPU 扫描直方图累积分裂点，返回最佳 bin 边界索引与增益。
func scanHistGainsCPU(histG, histH []float64, sumG, sumH, lambda float64) (bestSplit int, bestGain float64) {
	numBins := len(histG)
	if numBins < 2 {
		return -1, 0
	}
	var gLeft, hLeft float64
	for s := 0; s < numBins-1; s++ {
		gLeft += histG[s]
		hLeft += histH[s]
		gRight := sumG - gLeft
		hRight := sumH - hLeft
		if hLeft <= 0 || hRight <= 0 {
			continue
		}
		gain := splitGain(gLeft, hLeft, gRight, hRight, sumG, sumH, lambda)
		if gain > bestGain {
			bestGain = gain
			bestSplit = s
		}
	}
	return bestSplit, bestGain
}
