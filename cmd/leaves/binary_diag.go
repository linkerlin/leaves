package main

import (
	"math"
	"sort"

	"github.com/linkerlin/leaves/v2/metrics"
)

// binaryDiagnostics 二分类（binary:logistic）扩展诊断指标与最优 F1 阈值（WP-32）。
type binaryDiagnostics struct {
	AUC             float64        `json:"auc"`
	AccuracyAt05    float64        `json:"accuracy_at_05"`
	F1At05          float64        `json:"f1_at_05"`
	BestThreshold   float64        `json:"best_threshold"`
	BestF1          float64        `json:"best_f1"`
	BestPrecision   float64        `json:"best_precision"`
	BestRecall      float64        `json:"best_recall"`
	BestAccuracy    float64        `json:"best_accuracy"`
	ConfusionMatrix map[string]int `json:"confusion_matrix"`
}

// regressionDiagnostics 回归任务扩展诊断指标（WP-38 / DoD AO4）。
type regressionDiagnostics struct {
	R2           float64 `json:"r2"`
	MAE          float64 `json:"mae"`
	RMSE         float64 `json:"rmse"`
	MedAE        float64 `json:"medae"`
	P90AbsError  float64 `json:"p90_abs_error"`
	MaxAbsError  float64 `json:"max_abs_error"`
	MeanResidual float64 `json:"mean_residual"`
	OutlierRatio float64 `json:"outlier_ratio"`
}

// multiclassDiagnostics 多分类任务扩展诊断指标与混淆矩阵（WP-38 / DoD AO4）。
type multiclassDiagnostics struct {
	Accuracy        float64   `json:"accuracy"`
	MacroF1         float64   `json:"macro_f1"`
	MacroPrecision  float64   `json:"macro_precision"`
	MacroRecall     float64   `json:"macro_recall"`
	PerClassF1      []float64 `json:"per_class_f1"`
	ConfusionMatrix [][]int   `json:"confusion_matrix"`
}

func computeBinaryDiagnostics(labels, probs []float64) *binaryDiagnostics {
	n := len(labels)
	if n == 0 || len(probs) != n {
		return nil
	}
	aucVal, err := (metrics.AUC{}).Evaluate(labels, probs)
	if err != nil || math.IsNaN(aucVal) {
		aucVal = 0.5
	}

	evalAt := func(th float64) (tp, fp, tn, fn int, prec, rec, f1, acc float64) {
		for i := 0; i < n; i++ {
			y := labels[i] >= 0.5
			p := probs[i] >= th
			switch {
			case y && p:
				tp++
			case !y && p:
				fp++
			case !y && !p:
				tn++
			case y && !p:
				fn++
			}
		}
		if tp+fp > 0 {
			prec = float64(tp) / float64(tp+fp)
		}
		if tp+fn > 0 {
			rec = float64(tp) / float64(tp+fn)
		}
		if prec+rec > 0 {
			f1 = 2 * prec * rec / (prec + rec)
		}
		acc = float64(tp+tn) / float64(n)
		return
	}

	_, _, _, _, _, _, f105, acc05 := evalAt(0.5)

	bestTh := 0.5
	bestTP, bestFP, bestTN, bestFN, bestPrec, bestRec, bestF1, bestAcc := evalAt(0.5)

	for step := 5; step <= 95; step++ {
		th := float64(step) / 100.0
		tp, fp, tn, fn, prec, rec, f1, acc := evalAt(th)
		if f1 > bestF1+1e-9 || (math.Abs(f1-bestF1) <= 1e-9 && acc > bestAcc+1e-9) {
			bestTh = th
			bestTP, bestFP, bestTN, bestFN = tp, fp, tn, fn
			bestPrec, bestRec, bestF1, bestAcc = prec, rec, f1, acc
		}
	}

	return &binaryDiagnostics{
		AUC:           round4(aucVal),
		AccuracyAt05:  round4(acc05),
		F1At05:        round4(f105),
		BestThreshold: round4(bestTh),
		BestF1:        round4(bestF1),
		BestPrecision: round4(bestPrec),
		BestRecall:    round4(bestRec),
		BestAccuracy:  round4(bestAcc),
		ConfusionMatrix: map[string]int{
			"tp": bestTP,
			"fp": bestFP,
			"tn": bestTN,
			"fn": bestFN,
		},
	}
}

func computeRegressionDiagnostics(labels, preds []float64) *regressionDiagnostics {
	n := len(labels)
	if n == 0 || len(preds) != n {
		return nil
	}
	meanY := 0.0
	for _, y := range labels {
		meanY += y
	}
	meanY /= float64(n)

	absErrs := make([]float64, n)
	var sumAbs, sumSq, ssTot, sumRes float64
	for i := 0; i < n; i++ {
		res := labels[i] - preds[i]
		ae := math.Abs(res)
		absErrs[i] = ae
		sumAbs += ae
		sumSq += res * res
		sumRes += res
		dy := labels[i] - meanY
		ssTot += dy * dy
	}
	mae := sumAbs / float64(n)
	rmse := math.Sqrt(sumSq / float64(n))
	meanRes := sumRes / float64(n)
	r2 := 0.0
	if ssTot > 1e-12 {
		r2 = 1.0 - (sumSq / ssTot)
	}

	sort.Float64s(absErrs)
	medAE := absErrs[n/2]
	if n%2 == 0 && n >= 2 {
		medAE = 0.5 * (absErrs[n/2-1] + absErrs[n/2])
	}
	p90Idx := int(math.Floor(0.90 * float64(n-1)))
	if p90Idx < 0 {
		p90Idx = 0
	}
	p90AE := absErrs[p90Idx]
	maxAE := absErrs[n-1]

	outliers := 0
	if rmse > 1e-12 {
		thresh := 3.0 * rmse
		for _, ae := range absErrs {
			if ae > thresh {
				outliers++
			}
		}
	}

	return &regressionDiagnostics{
		R2:           round4(r2),
		MAE:          round4(mae),
		RMSE:         round4(rmse),
		MedAE:        round4(medAE),
		P90AbsError:  round4(p90AE),
		MaxAbsError:  round4(maxAE),
		MeanResidual: round4(meanRes),
		OutlierRatio: round4(float64(outliers) / float64(n)),
	}
}

func computeMulticlassDiagnostics(labels, rawMargins []float64, numClass int) *multiclassDiagnostics {
	n := len(labels)
	if n == 0 || numClass < 2 || len(rawMargins) < n*numClass {
		return nil
	}
	cm := make([][]int, numClass)
	for c := 0; c < numClass; c++ {
		cm[c] = make([]int, numClass)
	}
	correct := 0
	for i := 0; i < n; i++ {
		trueC := int(math.Round(labels[i]))
		predC := argmax(rawMargins[i*numClass : (i+1)*numClass])
		if trueC >= 0 && trueC < numClass && predC >= 0 && predC < numClass {
			cm[trueC][predC]++
		}
		if trueC == predC {
			correct++
		}
	}

	perClassF1 := make([]float64, numClass)
	var sumPrec, sumRec, sumF1 float64
	for c := 0; c < numClass; c++ {
		tp := cm[c][c]
		fp := 0
		fn := 0
		for k := 0; k < numClass; k++ {
			if k != c {
				fp += cm[k][c]
				fn += cm[c][k]
			}
		}
		var prec, rec, f1 float64
		if tp+fp > 0 {
			prec = float64(tp) / float64(tp+fp)
		}
		if tp+fn > 0 {
			rec = float64(tp) / float64(tp+fn)
		}
		if prec+rec > 0 {
			f1 = 2 * prec * rec / (prec + rec)
		}
		perClassF1[c] = round4(f1)
		sumPrec += prec
		sumRec += rec
		sumF1 += f1
	}

	kFloat := float64(numClass)
	return &multiclassDiagnostics{
		Accuracy:        round4(float64(correct) / float64(n)),
		MacroF1:         round4(sumF1 / kFloat),
		MacroPrecision:  round4(sumPrec / kFloat),
		MacroRecall:     round4(sumRec / kFloat),
		PerClassF1:      perClassF1,
		ConfusionMatrix: cm,
	}
}

// rankingDiagnostics 为排序学习任务（rank:ndcg/pairwise/listwise）提供多截断位与 Query 结构诊断（WP-42 / AA3）。
type rankingDiagnostics struct {
	NQueries            int     `json:"n_queries"`
	MeanGroupSize       float64 `json:"mean_group_size"`
	MinGroupSize        int     `json:"min_group_size"`
	MaxGroupSize        int     `json:"max_group_size"`
	ZeroRelevantQueries int     `json:"zero_relevant_queries"`
	NDCGAt1             float64 `json:"ndcg_at_1"`
	NDCGAt3             float64 `json:"ndcg_at_3"`
	NDCGAt5             float64 `json:"ndcg_at_5"`
	NDCGAt10            float64 `json:"ndcg_at_10"`
	MAP                 float64 `json:"map"`
	MRR                 float64 `json:"mrr"`
}

func computeRankingDiagnostics(labels, preds []float64, groups []int) *rankingDiagnostics {
	if len(groups) == 0 || len(labels) == 0 || len(labels) != len(preds) {
		return nil
	}
	nQ := len(groups)
	minG, maxG, totalRows, zeroRel := groups[0], groups[0], 0, 0
	var mrrSum float64
	offset := 0
	for _, gSize := range groups {
		if gSize <= 0 || offset+gSize > len(labels) {
			break
		}
		if gSize < minG {
			minG = gSize
		}
		if gSize > maxG {
			maxG = gSize
		}
		totalRows += gSize
		gLabels := labels[offset : offset+gSize]
		gPreds := preds[offset : offset+gSize]
		offset += gSize

		hasRel := false
		for _, y := range gLabels {
			if y > 0 {
				hasRel = true
				break
			}
		}
		if !hasRel {
			zeroRel++
			continue
		}
		order := make([]int, gSize)
		for i := range order {
			order[i] = i
		}
		sort.SliceStable(order, func(i, j int) bool {
			return gPreds[order[i]] > gPreds[order[j]]
		})
		for rankIdx, docIdx := range order {
			if gLabels[docIdx] > 0 {
				mrrSum += 1.0 / float64(rankIdx+1)
				break
			}
		}
	}

	evalNDCG := func(k int) float64 {
		m := metrics.NDCG{RankingMetric: metrics.RankingMetric{Groups: groups, K: k}}
		v, err := metrics.Evaluate(m, labels, preds, groups)
		if err != nil {
			return 0
		}
		return round4(v)
	}
	mapM := metrics.MAP{RankingMetric: metrics.RankingMetric{Groups: groups, K: 10}}
	mapVal, _ := metrics.Evaluate(mapM, labels, preds, groups)

	return &rankingDiagnostics{
		NQueries:            nQ,
		MeanGroupSize:       round4(float64(totalRows) / float64(nQ)),
		MinGroupSize:        minG,
		MaxGroupSize:        maxG,
		ZeroRelevantQueries: zeroRel,
		NDCGAt1:             evalNDCG(1),
		NDCGAt3:             evalNDCG(3),
		NDCGAt5:             evalNDCG(5),
		NDCGAt10:            evalNDCG(10),
		MAP:                 round4(mapVal),
		MRR:                 round4(mrrSum / float64(nQ)),
	}
}

func round4(v float64) float64 {
	return math.Round(v*10000) / 10000
}
