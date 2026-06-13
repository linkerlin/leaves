package metrics

import (
	"fmt"
	"math"
	"sort"
)

// Metric 评估指标接口。
type Metric interface {
	Name() string
	Evaluate(yTrue, yPred []float64) (float64, error)
	EvaluatePerGroup(yTrue, yPred []float64, groups []int) (float64, error)
	HigherIsBetter() bool
}

// EvaluatePerGroupSplit 将样本按 groups 切片后对点态指标取平均。
func EvaluatePerGroupSplit(m Metric, yTrue, yPred []float64, groups []int) (float64, error) {
	if len(groups) == 0 {
		return m.Evaluate(yTrue, yPred)
	}
	if sumInts(groups) != len(yTrue) || len(yPred) != len(yTrue) {
		return 0, fmt.Errorf("%s: per-group length mismatch", m.Name())
	}
	var total float64
	start := 0
	for _, g := range groups {
		if g <= 0 {
			return 0, fmt.Errorf("%s: invalid group size", m.Name())
		}
		end := start + g
		v, err := m.Evaluate(yTrue[start:end], yPred[start:end])
		if err != nil {
			return 0, err
		}
		total += v
		start = end
	}
	return total / float64(len(groups)), nil
}

// RMSE 均方根误差。
type RMSE struct{}

func (RMSE) Name() string           { return "rmse" }
func (RMSE) HigherIsBetter() bool   { return false }
func (RMSE) EvaluatePerGroup(yTrue, yPred []float64, groups []int) (float64, error) {
	return EvaluatePerGroupSplit(RMSE{}, yTrue, yPred, groups)
}
func (RMSE) Evaluate(yTrue, yPred []float64) (float64, error) {
	if len(yTrue) != len(yPred) || len(yTrue) == 0 {
		return 0, fmt.Errorf("rmse: length mismatch or empty")
	}
	var sum float64
	for i := range yTrue {
		d := yTrue[i] - yPred[i]
		sum += d * d
	}
	return math.Sqrt(sum / float64(len(yTrue))), nil
}

// MAE 平均绝对误差。
type MAE struct{}

func (MAE) Name() string           { return "mae" }
func (MAE) HigherIsBetter() bool   { return false }
func (MAE) EvaluatePerGroup(yTrue, yPred []float64, groups []int) (float64, error) {
	return EvaluatePerGroupSplit(MAE{}, yTrue, yPred, groups)
}
func (MAE) Evaluate(yTrue, yPred []float64) (float64, error) {
	if len(yTrue) != len(yPred) || len(yTrue) == 0 {
		return 0, fmt.Errorf("mae: length mismatch or empty")
	}
	var sum float64
	for i := range yTrue {
		sum += math.Abs(yTrue[i] - yPred[i])
	}
	return sum / float64(len(yTrue)), nil
}

// LogLoss 二分类对数损失（yTrue∈{0,1}，yPred 为概率）。
type LogLoss struct{}

func (LogLoss) Name() string         { return "logloss" }
func (LogLoss) HigherIsBetter() bool { return false }
func (LogLoss) EvaluatePerGroup(yTrue, yPred []float64, groups []int) (float64, error) {
	return EvaluatePerGroupSplit(LogLoss{}, yTrue, yPred, groups)
}
func (LogLoss) Evaluate(yTrue, yPred []float64) (float64, error) {
	if len(yTrue) != len(yPred) || len(yTrue) == 0 {
		return 0, fmt.Errorf("logloss: length mismatch or empty")
	}
	const eps = 1e-15
	var sum float64
	for i := range yTrue {
		p := clampProb(yPred[i], eps)
		y := yTrue[i]
		sum -= y*math.Log(p) + (1-y)*math.Log(1-p)
	}
	return sum / float64(len(yTrue)), nil
}

// Error 二分类错误率（yPred>0.5 为正类）。
type Error struct{}

func (Error) Name() string         { return "error" }
func (Error) HigherIsBetter() bool { return false }
func (Error) EvaluatePerGroup(yTrue, yPred []float64, groups []int) (float64, error) {
	return EvaluatePerGroupSplit(Error{}, yTrue, yPred, groups)
}
func (Error) Evaluate(yTrue, yPred []float64) (float64, error) {
	if len(yTrue) != len(yPred) || len(yTrue) == 0 {
		return 0, fmt.Errorf("error: length mismatch or empty")
	}
	wrong := 0
	for i := range yTrue {
		pred := 0.0
		if yPred[i] > 0.5 {
			pred = 1.0
		}
		if pred != yTrue[i] {
			wrong++
		}
	}
	return float64(wrong) / float64(len(yTrue)), nil
}

// AUC ROC-AUC（Wilcoxon-Mann-Whitney）。
type AUC struct{}

func (AUC) Name() string         { return "auc" }
func (AUC) HigherIsBetter() bool { return true }
func (AUC) EvaluatePerGroup(yTrue, yPred []float64, groups []int) (float64, error) {
	return EvaluatePerGroupSplit(AUC{}, yTrue, yPred, groups)
}
func (AUC) Evaluate(yTrue, yPred []float64) (float64, error) {
	if len(yTrue) != len(yPred) || len(yTrue) == 0 {
		return 0, fmt.Errorf("auc: length mismatch or empty")
	}
	type pair struct {
		score float64
		label float64
	}
	pairs := make([]pair, len(yTrue))
	pos, neg := 0, 0
	for i := range yTrue {
		pairs[i] = pair{yPred[i], yTrue[i]}
		if yTrue[i] > 0.5 {
			pos++
		} else {
			neg++
		}
	}
	if pos == 0 || neg == 0 {
		return 0, fmt.Errorf("auc: need both positive and negative labels")
	}
	sort.Slice(pairs, func(i, j int) bool { return pairs[i].score < pairs[j].score })
	rankSumPos := 0.0
	for i, p := range pairs {
		if p.label > 0.5 {
			rankSumPos += float64(i + 1)
		}
	}
	auc := (rankSumPos - float64(pos*(pos+1))/2) / float64(pos*neg)
	return auc, nil
}

// MLogLoss 多分类对数损失（yTrue 为类索引，yPred 为 [n*classes] 行优先概率）。
type MLogLoss struct {
	NumClass int
}

func (m MLogLoss) Name() string         { return "mlogloss" }
func (m MLogLoss) HigherIsBetter() bool { return false }
func (m MLogLoss) EvaluatePerGroup(yTrue, yPred []float64, groups []int) (float64, error) {
	return EvaluatePerGroupSplit(m, yTrue, yPred, groups)
}
func (m MLogLoss) Evaluate(yTrue, yPred []float64) (float64, error) {
	if m.NumClass <= 0 {
		return 0, fmt.Errorf("mlogloss: invalid num class")
	}
	n := len(yTrue)
	if n == 0 || len(yPred) != n*m.NumClass {
		return 0, fmt.Errorf("mlogloss: length mismatch")
	}
	const eps = 1e-15
	var sum float64
	for i := 0; i < n; i++ {
		c := int(yTrue[i])
		if c < 0 || c >= m.NumClass {
			return 0, fmt.Errorf("mlogloss: invalid label %d", c)
		}
		p := clampProb(yPred[i*m.NumClass+c], eps)
		sum -= math.Log(p)
	}
	return sum / float64(n), nil
}

func clampProb(p, eps float64) float64 {
	if p < eps {
		return eps
	}
	if p > 1-eps {
		return 1 - eps
	}
	return p
}
