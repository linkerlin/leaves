package leaves

import (
	"github.com/dmitryikh/leaves/data"
	"github.com/dmitryikh/leaves/io"
	"github.com/dmitryikh/leaves/train"
)

// TrainConfig 根包训练配置别名。
type TrainConfig = train.Config

// TrainLearner 根包训练器别名。
type TrainLearner = train.Learner

// NewTrainLearner 创建训练器。
func NewTrainLearner(cfg TrainConfig) (*TrainLearner, error) {
	return train.NewLearner(cfg)
}

// TrainDense 在 Dense 数据上训练并返回 Learner。
func TrainDense(vals []float64, rows, cols int, labels []float64, cfg TrainConfig) (*TrainLearner, error) {
	dm, err := data.NewDense(vals, rows, cols, labels, nil)
	if err != nil {
		return nil, err
	}
	learner, err := train.NewLearner(cfg)
	if err != nil {
		return nil, err
	}
	if err := learner.Fit(dm); err != nil {
		return nil, err
	}
	return learner, nil
}

// SaveTrainModel 保存训练模型为 leaves.json。
func SaveTrainModel(path string, learner *TrainLearner, objective string) error {
	if learner == nil {
		return io.ErrFormatNotImplemented("nil learner")
	}
	return learner.Save(path)
}
