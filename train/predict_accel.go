package train

import (
	"log"

	"github.com/linkerlin/leaves/booster"
	"github.com/linkerlin/leaves/data"
	"github.com/linkerlin/leaves/tree"
)

const (
	gpuMarginPredictMinRowsAuto   = 256
	gpuMarginPredictMinRowsWebGPU = 64
)

func gpuMarginPredictEnabled(l *Learner, nRow int) bool {
	if l == nil || l.booster == nil {
		return false
	}
	if _, ok := l.booster.(*booster.GBTree); !ok {
		return false
	}
	mode := l.effectiveAccelMode
	if mode == "" {
		mode = l.resolveEffectiveAccel(nRow)
	}
	switch mode {
	case AccelModeCPU:
		return false
	case AccelModeWebGPU:
		return nRow >= gpuMarginPredictMinRowsWebGPU && tree.BornWebGPUAvailable()
	default:
		return l.useGPUHist && nRow >= gpuMarginPredictMinRowsAuto && tree.BornWebGPUAvailable()
	}
}

func (l *Learner) gbtreeForest() *tree.ForestIR {
	gt, ok := l.booster.(*booster.GBTree)
	if !ok {
		return nil
	}
	return gt.Forest()
}

func (l *Learner) ensureMarginEngine(forest *tree.ForestIR) *tree.BornEngine {
	if l.marginEngine != nil {
		return l.marginEngine
	}
	if forest == nil || !tree.BornSupports(forest, tree.BackendBornGPU) {
		return nil
	}
	eng, err := tree.NewBornEngine(
		forest,
		tree.ApplyTransformRaw,
		tree.TransformRaw,
		l.numGroups,
		&tree.BornConfig{UseGPU: true},
	)
	if err != nil || eng == nil || !eng.BornUsingGPU() {
		if eng != nil {
			_ = eng.Close()
		}
		return nil
	}
	l.marginEngine = eng
	return eng
}

func (l *Learner) closeMarginEngine() {
	if l.marginEngine != nil {
		_ = l.marginEngine.Close()
		l.marginEngine = nil
	}
}

func (l *Learner) logMarginGPUOnce(n int) {
	if l.marginGPULogged {
		return
	}
	l.marginGPULogged = true
	log.Printf("[leaves/train] accel: gpu margin predict enabled rows=%d", n)
}

func matrixDenseVals(dm data.Matrix) (vals []float64, rows, cols int, ok bool) {
	if d, okDense := dm.(*data.Dense); okDense {
		return d.Data, d.Rows, d.Cols, true
	}
	rows = dm.NumRow()
	cols = dm.NumCol()
	vals = make([]float64, rows*cols)
	row := make([]float64, cols)
	for i := 0; i < rows; i++ {
		_ = dm.Row(i, row)
		off := i * cols
		copy(vals[off:off+cols], row)
	}
	return vals, rows, cols, true
}

// predictMarginsInternal 训练期 margin 预测；allowGPU 为 false 时始终 CPU（boosting 热路径）。
func (l *Learner) predictMarginsInternal(dm data.Matrix, out []float64, allowGPU bool) {
	n := dm.NumRow()
	if allowGPU && gpuMarginPredictEnabled(l, n) {
		if l.tryBornGPUPredict(dm, out, n) {
			return
		}
	}
	l.booster.PredictMargins(dm, out)
	l.marginPredictCPU++
}

func (l *Learner) tryBornGPUPredict(dm data.Matrix, out []float64, n int) (ok bool) {
	defer func() {
		if r := recover(); r != nil {
			l.closeMarginEngine()
			log.Printf("[leaves/train] accel: gpu margin predict error (%v), fallback cpu", r)
			ok = false
		}
	}()
	forest := l.gbtreeForest()
	eng := l.ensureMarginEngine(forest)
	if eng == nil {
		return false
	}
	vals, rows, cols, denseOK := matrixDenseVals(dm)
	if !denseOK {
		return false
	}
	if err := eng.PredictDense(vals, rows, cols, out, 0); err != nil {
		return false
	}
	l.marginPredictGPU++
	l.logMarginGPUOnce(n)
	return true
}
