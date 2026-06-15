package model

import (
	"time"

	"github.com/linkerlin/leaves/predict"
	"github.com/linkerlin/leaves/tree"
)

// PredictProfile PredictWithRequest 耗时统计。
type PredictProfile struct {
	Rows    int
	Elapsed time.Duration
}

// PredictWithProfile 执行预测并返回耗时（包装 tree.Profile 语义）。
func (e *Ensemble) PredictWithProfile(req predict.Request, out []float64) (PredictProfile, error) {
	var prof PredictProfile
	nRows, nCols, err := matrixShape(req.Matrix)
	if err != nil {
		return prof, err
	}
	prof.Rows = nRows
	start := time.Now()
	err = e.PredictWithRequest(req, out)
	prof.Elapsed = time.Since(start)
	if err != nil {
		return prof, err
	}
	if ne, ok := e.engine.(*tree.NativeEngine); ok && req.Output == predict.OutputValue {
		_ = ne // 预留：可扩展 WalkStats 聚合
		_, _ = nCols, ne
	}
	return prof, nil
}
