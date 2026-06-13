package booster

func gradHessAt(grad, hess []float64, row, class, numClass int) (float64, float64) {
	if numClass <= 1 {
		if row < len(grad) {
			return grad[row], hess[row]
		}
		return 0, 1
	}
	idx := row*numClass + class
	if idx < len(grad) {
		return grad[idx], hess[idx]
	}
	return 0, 1
}
