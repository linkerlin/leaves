package treebuilder

// scanHistGainsCPU 扫描直方图累积分裂点，返回最佳 bin 边界索引与增益。
// histG/histH 布局 [numBins*k]（bin 主序，每 bin 连续 k 类）；k 由 len(sumG) 推得。
func scanHistGainsCPU(histG, histH []float64, sumG, sumH []float64, lambda float64) (bestSplit int, bestGain float64) {
	k := len(sumG)
	if k < 1 {
		k = 1
	}
	numBins := len(histG) / k
	if numBins < 2 {
		return -1, 0
	}
	gLeft := make([]float64, k)
	hLeft := make([]float64, k)
	for s := 0; s < numBins-1; s++ {
		bk := s * k
		for c := 0; c < k; c++ {
			gLeft[c] += histG[bk+c]
			hLeft[c] += histH[bk+c]
		}
		gRight := make([]float64, k)
		hRight := make([]float64, k)
		for c := 0; c < k; c++ {
			gRight[c] = sumG[c] - gLeft[c]
			hRight[c] = sumH[c] - hLeft[c]
		}
		gain := splitGain(gLeft, hLeft, gRight, hRight, sumG, sumH, lambda)
		if gain > bestGain {
			bestGain = gain
			bestSplit = s
		}
	}
	return bestSplit, bestGain
}
