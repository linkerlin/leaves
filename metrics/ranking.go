package metrics

import (
	"fmt"
	"math"
	"sort"
)

// MAPE 平均绝对百分比误差（跳过 yTrue=0）。
type MAPE struct{}

func (MAPE) Name() string         { return "mape" }
func (MAPE) HigherIsBetter() bool { return false }
func (MAPE) EvaluatePerGroup(yTrue, yPred []float64, groups []int) (float64, error) {
	return EvaluatePerGroupSplit(MAPE{}, yTrue, yPred, groups)
}
func (MAPE) Evaluate(yTrue, yPred []float64) (float64, error) {
	if len(yTrue) != len(yPred) || len(yTrue) == 0 {
		return 0, fmt.Errorf("mape: length mismatch or empty")
	}
	var sum float64
	var n int
	for i := range yTrue {
		if yTrue[i] == 0 {
			continue
		}
		sum += math.Abs((yTrue[i] - yPred[i]) / yTrue[i])
		n++
	}
	if n == 0 {
		return 0, fmt.Errorf("mape: no non-zero labels")
	}
	return sum / float64(n), nil
}

// RMSLE 均方对数误差（要求 yTrue,yPred≥0）。
type RMSLE struct{}

func (RMSLE) Name() string         { return "rmsle" }
func (RMSLE) HigherIsBetter() bool { return false }
func (RMSLE) EvaluatePerGroup(yTrue, yPred []float64, groups []int) (float64, error) {
	return EvaluatePerGroupSplit(RMSLE{}, yTrue, yPred, groups)
}
func (RMSLE) Evaluate(yTrue, yPred []float64) (float64, error) {
	if len(yTrue) != len(yPred) || len(yTrue) == 0 {
		return 0, fmt.Errorf("rmsle: length mismatch or empty")
	}
	var sum float64
	for i := range yTrue {
		if yTrue[i] < 0 || yPred[i] < 0 {
			return 0, fmt.Errorf("rmsle: negative value at %d", i)
		}
		lt := math.Log1p(yTrue[i])
		lp := math.Log1p(yPred[i])
		d := lt - lp
		sum += d * d
	}
	return math.Sqrt(sum / float64(len(yTrue))), nil
}

// MError 多分类错误率（yTrue 为类索引，yPred 为行优先概率）。
type MError struct {
	NumClass int
}

func (m MError) Name() string         { return "merror" }
func (m MError) HigherIsBetter() bool { return false }
func (m MError) EvaluatePerGroup(yTrue, yPred []float64, groups []int) (float64, error) {
	return EvaluatePerGroupSplit(m, yTrue, yPred, groups)
}
func (m MError) Evaluate(yTrue, yPred []float64) (float64, error) {
	if m.NumClass <= 0 {
		return 0, fmt.Errorf("merror: invalid num class")
	}
	n := len(yTrue)
	if n == 0 || len(yPred) != n*m.NumClass {
		return 0, fmt.Errorf("merror: length mismatch")
	}
	wrong := 0
	for i := 0; i < n; i++ {
		trueClass := int(yTrue[i])
		predClass := 0
		best := yPred[i*m.NumClass]
		for c := 1; c < m.NumClass; c++ {
			if yPred[i*m.NumClass+c] > best {
				best = yPred[i*m.NumClass+c]
				predClass = c
			}
		}
		if predClass != trueClass {
			wrong++
		}
	}
	return float64(wrong) / float64(n), nil
}

// RankingMetric 排序指标公共输入。
type RankingMetric struct {
	Groups []int // 每个 query 的样本数，和为 len(yTrue)
	K      int   // NDCG@k 的 k；0 表示全量
}

// NDCG 归一化折损累积增益（yTrue 为相关性，yPred 为分数）。
type NDCG struct {
	RankingMetric
}

func (m NDCG) Name() string         { return "ndcg" }
func (m NDCG) HigherIsBetter() bool { return true }
func (m NDCG) EvaluatePerGroup(yTrue, yPred []float64, groups []int) (float64, error) {
	if len(groups) == 0 {
		return m.Evaluate(yTrue, yPred)
	}
	return evalRanking(yTrue, yPred, groups, m.K, true)
}
func (m NDCG) Evaluate(yTrue, yPred []float64) (float64, error) {
	return evalRanking(yTrue, yPred, m.Groups, m.K, true)
}

// MAP 平均精度（二值相关性）。
type MAP struct {
	RankingMetric
}

func (m MAP) Name() string         { return "map" }
func (m MAP) HigherIsBetter() bool { return true }
func (m MAP) EvaluatePerGroup(yTrue, yPred []float64, groups []int) (float64, error) {
	if len(groups) == 0 {
		return m.Evaluate(yTrue, yPred)
	}
	return evalRanking(yTrue, yPred, groups, m.K, false)
}
func (m MAP) Evaluate(yTrue, yPred []float64) (float64, error) {
	return evalRanking(yTrue, yPred, m.Groups, m.K, false)
}

func evalRanking(yTrue, yPred []float64, groups []int, k int, ndcg bool) (float64, error) {
	if len(groups) == 0 {
		return 0, fmt.Errorf("ranking: empty groups")
	}
	if sumInts(groups) != len(yTrue) || len(yPred) != len(yTrue) {
		return 0, fmt.Errorf("ranking: length mismatch")
	}
	var total float64
	start := 0
	for _, g := range groups {
		if g <= 0 {
			return 0, fmt.Errorf("ranking: invalid group size")
		}
		end := start + g
		score, err := evalOneGroup(yTrue[start:end], yPred[start:end], k, ndcg)
		if err != nil {
			return 0, err
		}
		total += score
		start = end
	}
	return total / float64(len(groups)), nil
}

func evalOneGroup(yTrue, yPred []float64, k int, ndcg bool) (float64, error) {
	n := len(yTrue)
	if k <= 0 || k > n {
		k = n
	}
	type pair struct {
		label float64
		score float64
	}
	pairs := make([]pair, n)
	for i := range yTrue {
		pairs[i] = pair{yTrue[i], yPred[i]}
	}
	sort.Slice(pairs, func(i, j int) bool { return pairs[i].score > pairs[j].score })

	if ndcg {
		dcg := 0.0
		for i := 0; i < k; i++ {
			dcg += (math.Pow(2, pairs[i].label) - 1) / math.Log2(float64(i)+2)
		}
		sortedLabels := make([]float64, n)
		copy(sortedLabels, yTrue)
		sort.Slice(sortedLabels, func(i, j int) bool { return sortedLabels[i] > sortedLabels[j] })
		idcg := 0.0
		for i := 0; i < k; i++ {
			idcg += (math.Pow(2, sortedLabels[i]) - 1) / math.Log2(float64(i)+2)
		}
		if idcg == 0 {
			return 0, nil
		}
		return dcg / idcg, nil
	}

	hits := 0
	precSum := 0.0
	relevant := 0
	for _, y := range yTrue {
		if y > 0.5 {
			relevant++
		}
	}
	if relevant == 0 {
		return 0, nil
	}
	for i := 0; i < k; i++ {
		if pairs[i].label > 0.5 {
			hits++
			precSum += float64(hits) / float64(i+1)
		}
	}
	return precSum / float64(relevant), nil
}

func sumInts(a []int) int {
	s := 0
	for _, v := range a {
		s += v
	}
	return s
}

// All 返回内置指标列表（便于注册/文档）。
func All() []Metric {
	return []Metric{
		RMSE{}, MAE{}, MAPE{}, RMSLE{},
		LogLoss{}, Error{}, AUC{},
		MLogLoss{NumClass: 2},
		MError{NumClass: 2},
	}
}
