package rankutil

import (
	"github.com/linkerlin/leaves/v2/data"
	recrank "github.com/linkerlin/leaves/v2/recsys/rankutil"
	"github.com/linkerlin/leaves/v2/train"
)

// NDCGAtK 委托 recsys/rankutil。
func NDCGAtK(dm data.Matrix, preds []float64, k int) (float64, error) {
	return recrank.NDCGAtK(dm, preds, k)
}

// PredictMargins 委托 recsys/rankutil。
func PredictMargins(learner *train.Learner, dm data.Matrix) ([]float64, error) {
	return recrank.PredictMargins(learner, dm)
}
