package train

import (
	"fmt"
	"math/rand"

	"github.com/dmitryikh/leaves/data"
)

// CVResult 交叉验证结果。
type CVResult struct {
	FoldMetrics []float64
	MeanMetric  float64
	StdMetric   float64
}

// CrossValidate K 折交叉验证（回归/二分类 metric）。
func CrossValidate(cfg Config, dm data.Matrix, folds int) (*CVResult, error) {
	if dm == nil {
		return nil, fmt.Errorf("train: nil matrix")
	}
	if folds < 2 {
		folds = 5
	}
	n := dm.NumRow()
	if n < folds {
		return nil, fmt.Errorf("train: not enough rows %d for %d folds", n, folds)
	}
	indices := make([]int, n)
	for i := range indices {
		indices[i] = i
	}
	rng := rand.New(rand.NewSource(cfg.Seed))
	rng.Shuffle(n, func(i, j int) { indices[i], indices[j] = indices[j], indices[i] })

	foldSize := n / folds
	metrics := make([]float64, 0, folds)
	learnerCfg := cfg
	learnerCfg.EarlyStop = nil
	learnerCfg.EvalSet = nil

	for f := 0; f < folds; f++ {
		start := f * foldSize
		end := start + foldSize
		if f == folds-1 {
			end = n
		}
		valIdx := indices[start:end]
		trainIdx := append([]int(nil), indices[:start]...)
		trainIdx = append(trainIdx, indices[end:]...)

		trainDM, err := data.SliceMatrix(dm, trainIdx)
		if err != nil {
			return nil, err
		}
		valDM, err := data.SliceMatrix(dm, valIdx)
		if err != nil {
			return nil, err
		}

		learner, err := NewLearner(learnerCfg)
		if err != nil {
			return nil, err
		}
		if err := learner.Fit(trainDM); err != nil {
			return nil, err
		}
		score, err := evalMetricOnSet(learner, valDM)
		if err != nil {
			return nil, err
		}
		metrics = append(metrics, score)
	}

	var sum, sq float64
	for _, m := range metrics {
		sum += m
		sq += m * m
	}
	mean := sum / float64(len(metrics))
	variance := sq/float64(len(metrics)) - mean*mean
	if variance < 0 {
		variance = 0
	}
	return &CVResult{
		FoldMetrics: metrics,
		MeanMetric:  mean,
		StdMetric:   sqrt(variance),
	}, nil
}

func sqrt(x float64) float64 {
	if x <= 0 {
		return 0
	}
	z := x
	for i := 0; i < 10; i++ {
		z -= (z*z - x) / (2 * z)
	}
	return z
}
