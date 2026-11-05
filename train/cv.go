package train

import (
	"fmt"
	"math/rand"
	"sort"
	"strings"

	"github.com/linkerlin/leaves/v2/data"
)

// CVResult 交叉验证结果。
type CVResult struct {
	FoldMetrics    []float64
	MeanMetric     float64
	StdMetric      float64
	FoldBestRounds []int
	MeanBestRound  float64
}

// CrossValidate K 折交叉验证（支持回归 KFold、分类 StratifiedKFold、排序 GroupKFold 及折内早停）。
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
	rng := rand.New(rand.NewSource(cfg.Seed))

	gm, isGrouped := dm.(data.GroupedMatrix)
	var foldTrainSets []data.Matrix
	var foldValSets []data.Matrix

	if isGrouped {
		groups := gm.Groups()
		ng := len(groups)
		if ng < folds {
			return nil, fmt.Errorf("train: not enough groups %d for %d folds", ng, folds)
		}
		gIdx := make([]int, ng)
		for i := range gIdx {
			gIdx[i] = i
		}
		rng.Shuffle(ng, func(i, j int) { gIdx[i], gIdx[j] = gIdx[j], gIdx[i] })
		foldSize := ng / folds
		for f := 0; f < folds; f++ {
			start := f * foldSize
			end := start + foldSize
			if f == folds-1 {
				end = ng
			}
			valG := append([]int(nil), gIdx[start:end]...)
			trainG := append([]int(nil), gIdx[:start]...)
			trainG = append(trainG, gIdx[end:]...)
			sort.Ints(valG)
			sort.Ints(trainG)
			trDM, err := data.SliceMatrixGroups(gm, trainG)
			if err != nil {
				return nil, err
			}
			vaDM, err := data.SliceMatrixGroups(gm, valG)
			if err != nil {
				return nil, err
			}
			foldTrainSets = append(foldTrainSets, trDM)
			foldValSets = append(foldValSets, vaDM)
		}
	} else {
		var foldValIndices [][]int
		if isClassificationObjective(cfg.Objective) {
			foldValIndices = buildStratifiedFolds(dm.Labels(), folds, rng)
		} else {
			indices := make([]int, n)
			for i := range indices {
				indices[i] = i
			}
			rng.Shuffle(n, func(i, j int) { indices[i], indices[j] = indices[j], indices[i] })
			foldSize := n / folds
			foldValIndices = make([][]int, folds)
			for f := 0; f < folds; f++ {
				start := f * foldSize
				end := start + foldSize
				if f == folds-1 {
					end = n
				}
				foldValIndices[f] = indices[start:end]
			}
		}
		for f := 0; f < folds; f++ {
			valIdx := foldValIndices[f]
			valSet := make(map[int]bool, len(valIdx))
			for _, vi := range valIdx {
				valSet[vi] = true
			}
			trainIdx := make([]int, 0, n-len(valIdx))
			for f2 := 0; f2 < folds; f2++ {
				if f2 == f {
					continue
				}
				trainIdx = append(trainIdx, foldValIndices[f2]...)
			}
			if len(trainIdx) == 0 {
				for i := 0; i < n; i++ {
					if !valSet[i] {
						trainIdx = append(trainIdx, i)
					}
				}
			}
			trDM, err := data.SliceMatrix(dm, trainIdx)
			if err != nil {
				return nil, err
			}
			vaDM, err := data.SliceMatrix(dm, valIdx)
			if err != nil {
				return nil, err
			}
			foldTrainSets = append(foldTrainSets, trDM)
			foldValSets = append(foldValSets, vaDM)
		}
	}

	metrics := make([]float64, 0, folds)
	var foldBestRounds []int
	hasEarlyStop := cfg.EarlyStop != nil && cfg.EarlyStop.Rounds > 0

	for f := 0; f < folds; f++ {
		trainDM := foldTrainSets[f]
		valDM := foldValSets[f]

		learnerCfg := cfg
		learnerCfg.EvalSet = nil
		learnerCfg.EarlyStop = nil
		var foldES *EarlyStopping
		if hasEarlyStop {
			foldES = NewEarlyStopping(cfg.EarlyStop.Rounds, cfg.EarlyStop.Maximize)
			learnerCfg.EvalSet = valDM
			learnerCfg.EarlyStop = foldES
		}

		learner, err := NewLearner(learnerCfg)
		if err != nil {
			return nil, err
		}
		if err := learner.Fit(trainDM); err != nil {
			return nil, err
		}
		if hasEarlyStop {
			learner.ApplyBestRound()
			br := foldES.BestRound()
			if br <= 0 {
				br = learner.BoostRounds()
			}
			foldBestRounds = append(foldBestRounds, br)
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
	var meanBestRound float64
	if len(foldBestRounds) > 0 {
		var bsum int
		for _, b := range foldBestRounds {
			bsum += b
		}
		meanBestRound = float64(bsum) / float64(len(foldBestRounds))
	}
	return &CVResult{
		FoldMetrics:    metrics,
		MeanMetric:     mean,
		StdMetric:      sqrt(variance),
		FoldBestRounds: foldBestRounds,
		MeanBestRound:  meanBestRound,
	}, nil
}

func isClassificationObjective(obj string) bool {
	s := strings.ToLower(strings.TrimSpace(obj))
	return strings.HasPrefix(s, "binary:") || strings.HasPrefix(s, "multi:")
}

func buildStratifiedFolds(labels []float64, folds int, rng *rand.Rand) [][]int {
	buckets := make(map[int][]int)
	var keys []int
	for i, y := range labels {
		k := int(y)
		if _, ok := buckets[k]; !ok {
			keys = append(keys, k)
		}
		buckets[k] = append(buckets[k], i)
	}
	sort.Ints(keys)
	out := make([][]int, folds)
	rr := 0
	for _, k := range keys {
		b := buckets[k]
		rng.Shuffle(len(b), func(i, j int) { b[i], b[j] = b[j], b[i] })
		for _, idx := range b {
			out[rr%folds] = append(out[rr%folds], idx)
			rr++
		}
	}
	return out
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
