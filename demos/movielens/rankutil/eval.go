package rankutil

import (
	"fmt"

	"github.com/linkerlin/leaves/data"
	"github.com/linkerlin/leaves/metrics"
	"github.com/linkerlin/leaves/train"
)

// NDCGAtK 在带 groups 的矩阵上计算 NDCG@k。
func NDCGAtK(dm data.Matrix, preds []float64, k int) (float64, error) {
	groups, err := data.GroupsFromRanking(dm)
	if err != nil {
		return 0, err
	}
	name := "ndcg"
	opt := metrics.Options{Groups: groups}
	if k > 0 {
		name = fmt.Sprintf("ndcg@%d", k)
		opt.NDCGK = k
	}
	m, err := metrics.Resolve(name, opt)
	if err != nil {
		return 0, err
	}
	return m.Evaluate(dm.Labels(), preds)
}

// PredictMargins 用已训练 Learner 在矩阵上预测 margin。
func PredictMargins(learner *train.Learner, dm data.Matrix) ([]float64, error) {
	out := make([]float64, dm.NumRow())
	if err := learner.PredictMargins(dm, out); err != nil {
		return nil, err
	}
	return out, nil
}
