package explain

import "github.com/linkerlin/leaves/tree"

// ImportanceType 特征重要性类型。
type ImportanceType int

const (
	// ImportanceWeight 按分裂次数统计（weight）。
	ImportanceWeight ImportanceType = iota
	// ImportanceGain 按分裂增益求和（需 TreeIR.SplitGain）。
	ImportanceGain
	// ImportanceTotalGain 增益占比（归一化到 1）。
	ImportanceTotalGain
	// ImportanceCover 按节点覆盖率求和（需 TreeIR.SumHess）。
	ImportanceCover
	// ImportanceTotalCover 覆盖率占比（归一化到 1）。
	ImportanceTotalCover
)

// FeatureImportance 特征重要性结果。
type FeatureImportance struct {
	Scores []float64
	Names  []string
}

// ComputeImportance 计算森林级特征重要性。
func ComputeImportance(f *tree.ForestIR, kind ImportanceType, featureNames []string) *FeatureImportance {
	if f == nil {
		return nil
	}
	n := f.NumFeatures
	if n <= 0 {
		n = maxFeatureIndex(f) + 1
	}
	scores := make([]float64, n)
	for i := range f.Trees {
		accumulateTree(&f.Trees[i], kind, scores)
	}
	normalizeImportance(kind, scores)
	names := featureNames
	if len(names) < n {
		names = make([]string, n)
		for i := range names {
			names[i] = featureNameOrDefault(featureNames, i)
		}
	}
	return &FeatureImportance{Scores: scores, Names: names[:n]}
}

func accumulateTree(t *tree.TreeIR, kind ImportanceType, scores []float64) {
	for i := 0; i < t.NumNodes; i++ {
		feat := int(t.SplitFeature[i])
		if feat < 0 || feat >= len(scores) {
			continue
		}
		switch kind {
		case ImportanceGain, ImportanceTotalGain:
			if i < len(t.SplitGain) {
				scores[feat] += t.SplitGain[i]
			} else {
				scores[feat]++
			}
		case ImportanceCover, ImportanceTotalCover:
			if i < len(t.SumHess) {
				scores[feat] += t.SumHess[i]
			} else {
				scores[feat]++
			}
		default:
			scores[feat]++
		}
	}
}

func normalizeImportance(kind ImportanceType, scores []float64) {
	if kind != ImportanceTotalGain && kind != ImportanceTotalCover {
		return
	}
	total := 0.0
	for _, s := range scores {
		total += s
	}
	if total <= 0 {
		return
	}
	for i := range scores {
		scores[i] /= total
	}
}

func maxFeatureIndex(f *tree.ForestIR) int {
	max := 0
	for i := range f.Trees {
		t := &f.Trees[i]
		for j := 0; j < t.NumNodes; j++ {
			if int(t.SplitFeature[j]) > max {
				max = int(t.SplitFeature[j])
			}
		}
	}
	return max
}

func featureNameOrDefault(names []string, i int) string {
	if i < len(names) && names[i] != "" {
		return names[i]
	}
	return "f" + itoa(i)
}

func itoa(i int) string {
	if i == 0 {
		return "0"
	}
	var buf [20]byte
	pos := len(buf)
	n := i
	for n > 0 {
		pos--
		buf[pos] = byte('0' + n%10)
		n /= 10
	}
	return string(buf[pos:])
}
