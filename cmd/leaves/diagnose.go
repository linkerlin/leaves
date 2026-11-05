package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"sort"
	"strconv"
	"strings"

	"github.com/linkerlin/leaves/v2/explain"
	leavesio "github.com/linkerlin/leaves/v2/io"
)

type roundLogItem struct {
	Round int      `json:"round"`
	LR    float64  `json:"lr"`
	Train *float64 `json:"train"`
	Val   *float64 `json:"val"`
}

func cmdDiagnose(args []string) error {
	fs := flag.NewFlagSet("diagnose", flag.ContinueOnError)
	fs.Usage = func() {
		fmt.Fprintln(fs.Output(), "用法: leaves diagnose [--model PATH] [--rounds-log|--rounds rounds.jsonl] [--metrics metrics.json] [--data train.csv] [--out PATH]")
	}
	modelPath := fs.String("model", "", "模型路径（计算特征增益与零增益特征，WP-35）")
	roundsLog := fs.String("rounds-log", "", "train --emit-rounds 产出的逐轮日志 JSONL")
	roundsAlias := fs.String("rounds", "", "--rounds-log 别名（train --emit-rounds 产出的逐轮日志 JSONL）")
	metricsIn := fs.String("metrics", "", "train 产出的 metrics.json")
	dataPath := fs.String("data", "", "数据路径（联动 sniff 检查常数列、ID 列、缺失值与类别列，WP-35）")
	labelCol := fs.String("label-col", "", "显式指定标签列名或 0-based 索引（配合 --data 使用）")
	dropCols := fs.String("drop-cols", "", "逗号分隔已丢弃列名或索引（配合 --data 使用）")
	outPath := fs.String("out", "", "诊断报告 JSON 输出路径（空=stdout）")
	if err := fs.Parse(args); err != nil {
		return err
	}
	rPath := *roundsLog
	if rPath == "" {
		rPath = *roundsAlias
	}
	if rPath == "" && *metricsIn == "" && *modelPath == "" && *dataPath == "" {
		return errUsage("leaves diagnose 至少需要 --model、--rounds-log（或 --rounds）、--metrics、--data 之一")
	}

	var mdoc metricsDoc
	hasMetrics := false
	if *metricsIn != "" {
		b, err := os.ReadFile(*metricsIn)
		if err != nil {
			return errAgentWrap("data_load", fmt.Sprintf("read --metrics: %v", err),
				"检查 metrics.json 路径", false, err)
		}
		if err := json.Unmarshal(b, &mdoc); err != nil {
			return errAgentWrap("data_load", fmt.Sprintf("parse --metrics: %v", err),
				"确认 JSON 格式有效", false, err)
		}
		hasMetrics = true
	}

	var items []roundLogItem
	if rPath != "" {
		f, err := os.Open(rPath)
		if err != nil {
			return errAgentWrap("data_load", fmt.Sprintf("open --rounds-log: %v", err),
				"检查 rounds.jsonl 路径（由 train --emit-rounds 生成）", false, err)
		}
		defer f.Close()
		sc := bufio.NewScanner(f)
		for sc.Scan() {
			line := strings.TrimSpace(sc.Text())
			if line == "" {
				continue
			}
			var it roundLogItem
			if err := json.Unmarshal([]byte(line), &it); err != nil {
				return errAgent("data_load", fmt.Sprintf("rounds.jsonl 解析失败: %v", err),
					"确认文件由 leaves train --emit-rounds 生成", false)
			}
			items = append(items, it)
		}
	}

	maximize := mdoc.Maximize
	if !hasMetrics && len(items) >= 2 && items[0].Train != nil && items[len(items)-1].Train != nil {
		// 若无 metrics.json，从训练指标趋势推断：train 随轮数上升则为 maximize（如 auc/ndcg）
		if *items[len(items)-1].Train > *items[0].Train {
			maximize = true
		}
	}

	diagnosis := "healthy"
	var secondaryDiagnoses []string
	var suggestions []string
	var cliArgs []string
	var bestRound int
	var bestVal, finalTrain, finalVal, overfitGap float64
	hasVal := false

	if len(items) > 0 {
		last := items[len(items)-1]
		if last.Train != nil {
			finalTrain = *last.Train
		}
		bestIdx := -1
		signFlips := 0
		prevDelta := 0.0
		for i, it := range items {
			if it.Val != nil {
				hasVal = true
				v := *it.Val
				finalVal = v
				if bestIdx < 0 {
					bestIdx = i
					bestVal = v
					bestRound = it.Round + 1
				} else {
					better := (!maximize && v < bestVal) || (maximize && v > bestVal)
					if better {
						bestIdx = i
						bestVal = v
						bestRound = it.Round + 1
					}
				}
				if i > 0 && items[i-1].Val != nil {
					d := v - *items[i-1].Val
					if prevDelta != 0 && ((d > 0 && prevDelta < 0) || (d < 0 && prevDelta > 0)) {
						signFlips++
					}
					if d != 0 {
						prevDelta = d
					}
				}
			}
		}

		if hasVal {
			if maximize {
				overfitGap = finalTrain - finalVal
			} else {
				overfitGap = finalVal - finalTrain
			}
			valDegraded := (!maximize && finalVal > bestVal*1.02) || (maximize && finalVal < bestVal*0.98)
			earlyPeak := bestRound > 0 && bestRound <= (len(items)*2)/3
			switch {
			case earlyPeak && (valDegraded || overfitGap > math.Abs(bestVal)*0.15):
				diagnosis = "overfitting"
				suggestions = []string{
					fmt.Sprintf("enable --early-stop (peak at round %d of %d)", bestRound, len(items)),
					"reduce --depth (e.g. -1 or -2) or increase --min-child-weight / --lambda",
					"set --subsample 0.8 --colsample 0.8",
				}
			case len(items) >= 6 && signFlips >= (len(items)*2)/3:
				diagnosis = "oscillating"
				suggestions = []string{
					"reduce --lr (e.g. halve learning rate) and increase --rounds proportionally",
					"increase --min-child-weight to stabilize leaf weights",
				}
			case bestRound >= len(items)-1 && overfitGap < math.Abs(bestVal)*0.05:
				// 直到最后一轮仍在改善且 train/val 差距极小 → 欠拟合，可继续加轮数或深度
				diagnosis = "underfitting"
				suggestions = []string{
					"increase --rounds (model still improving at final round)",
					"increase --depth or reduce --lambda / --min-child-weight",
				}
			}
		} else if len(items) >= 4 && items[0].Train != nil && last.Train != nil {
			initT := *items[0].Train
			relImp := math.Abs(finalTrain-initT) / math.Max(math.Abs(initT), 1e-9)
			if relImp < 0.01 {
				diagnosis = "underfitting"
				suggestions = []string{
					"increase --lr or --depth (training metric barely moved)",
					"check feature quality with leaves sniff",
				}
			}
		}
	}

	if hasMetrics && diagnosis == "healthy" {
		if mdoc.BestRound > 0 && mdoc.StoppedRound > mdoc.BestRound {
			diagnosis = "overfitting"
			bestRound = mdoc.BestRound
			suggestions = []string{
				fmt.Sprintf("use best_round=%d for full-data finalization", mdoc.BestRound),
				"increase regularization (--lambda, --min-child-weight) or lower --depth",
			}
		} else if mdoc.CVFolds >= 2 && mdoc.CVMean != 0 && mdoc.CVStd/math.Max(math.Abs(mdoc.CVMean), 1e-9) > 0.15 {
			diagnosis = "high_variance"
			suggestions = []string{
				"high fold variance (cv_std / cv_mean > 15%): lower --depth or increase --min-child-weight",
				"use leaves runs pareto for fold-level non-dominated parent selection",
			}
		}
	}

	// 根据主诊断生成结构化 CLI 参数建议
	cliArgs = buildDiagnoseCLIArgs(diagnosis, bestRound, mdoc.Params)

	// WP-35：联动 --data Sniff 与 --model 特征增益重要性
	var sniffFeatureNames []string
	var dataWarnings []string
	var constantFeatures []string
	var idLikeCols []string
	var categoricalCols []string
	var suggestedScalePosWeight float64
	dropSet := map[string]bool{}
	var suggestedDropCols []string
	addDropCol := func(c string) {
		c = strings.TrimSpace(c)
		if c == "" || dropSet[c] {
			return
		}
		dropSet[c] = true
		suggestedDropCols = append(suggestedDropCols, c)
	}

	if *dataPath != "" {
		effLabel := *labelCol
		effDrops := *dropCols
		if hasMetrics && mdoc.Params != nil {
			if effLabel == "" {
				effLabel = mdoc.Params.LabelCol
			}
			if effDrops == "" {
				effDrops = mdoc.Params.DropCols
			}
		}
		rep, err := buildSniffReport(*dataPath, "nan", effLabel, effDrops, true)
		if err != nil {
			return err
		}
		if fn, ok := rep["feature_names"].([]string); ok {
			sniffFeatureNames = fn
		}
		if ids, ok := rep["id_like_columns"].([]string); ok {
			idLikeCols = ids
		}
		if cats, ok := rep["categorical_columns"].([]string); ok {
			categoricalCols = cats
		}
		if sdrops, ok := rep["suggested_drop_cols"].([]string); ok {
			for _, c := range sdrops {
				addDropCol(c)
			}
		}
		if dq, ok := rep["data_quality"].(map[string]any); ok && dq != nil {
			if w, ok := dq["warnings"].([]string); ok {
				dataWarnings = w
			}
			if cf, ok := dq["constant_features"].([]string); ok {
				constantFeatures = cf
				for _, c := range cf {
					addDropCol(c)
				}
			}
			if nCells, ok := dq["nan_cells"].(int); ok && nCells > 0 {
				suggestions = append(suggestions, fmt.Sprintf("data contains %d missing cells: pass --na-policy nan or --na-policy skip-row", nCells))
				if !hasMetrics || mdoc.Params == nil || mdoc.Params.NAPolicy == "" || mdoc.Params.NAPolicy == "error" {
					cliArgs = append(cliArgs, "--na-policy", "nan")
				}
			}
			if posRatio, ok := asFloat(dq["label_pos_ratio"]); ok && posRatio > 0 && posRatio < 1 {
				if posRatio < 0.25 || posRatio > 0.75 {
					spw := math.Round(((1.0-posRatio)/posRatio)*100) / 100
					if spw > 0 && spw != 1.0 {
						suggestedScalePosWeight = spw
						suggestions = append(suggestions, fmt.Sprintf("imbalanced binary label (pos_ratio=%.3f): pass --scale-pos-weight %.2f", posRatio, spw))
						if !hasMetrics || mdoc.Params == nil || mdoc.Params.ScalePosWeight <= 0 {
							cliArgs = append(cliArgs, "--scale-pos-weight", strconv.FormatFloat(spw, 'f', -1, 64))
						}
					}
				}
			}
		}
		if len(categoricalCols) > 0 {
			suggestions = append(suggestions, fmt.Sprintf("categorical columns detected (%s): pass --auto-encode", strings.Join(categoricalCols, ",")))
			if !hasMetrics || mdoc.Params == nil || !mdoc.Params.AutoEncode {
				cliArgs = append(cliArgs, "--auto-encode")
			}
		}
	}

	var zeroGainFeatures []string
	var topFeatures []map[string]any
	if *modelPath != "" {
		ir, _, _, err := leavesio.LoadModelIR(*modelPath)
		if err != nil {
			return errAgentWrap("model_load", fmt.Sprintf("load --model %s: %v", *modelPath, err),
				"确认模型文件存在且为受支持格式", false, err)
		}
		if ir.Forest != nil {
			names := ir.FeatureNames
			if len(names) == 0 && len(sniffFeatureNames) > 0 {
				names = sniffFeatureNames
			}
			fi := explain.ComputeImportance(ir.Forest, explain.ImportanceGain, names)
			if fi != nil {
				totalGain := 0.0
				for _, s := range fi.Scores {
					totalGain += s
				}
				type featScore struct {
					name  string
					score float64
				}
				var ranked []featScore
				for i, s := range fi.Scores {
					nm := fi.Names[i]
					if s <= 0 {
						zeroGainFeatures = append(zeroGainFeatures, nm)
						addDropCol(nm)
					}
					ranked = append(ranked, featScore{name: nm, score: s})
				}
				sort.SliceStable(ranked, func(i, j int) bool {
					return ranked[i].score > ranked[j].score
				})
				limit := len(ranked)
				if limit > 10 {
					limit = 10
				}
				for i := 0; i < limit; i++ {
					pct := 0.0
					if totalGain > 0 {
						pct = ranked[i].score / totalGain
					}
					topFeatures = append(topFeatures, map[string]any{
						"name":     ranked[i].name,
						"gain":     ranked[i].score,
						"gain_pct": pct,
					})
				}
			}
		}
		if len(zeroGainFeatures) > 0 {
			if diagnosis == "healthy" {
				diagnosis = "feature_noise"
			} else {
				secondaryDiagnoses = append(secondaryDiagnoses, "feature_noise")
			}
			suggestions = append(suggestions, fmt.Sprintf("zero-gain features detected (%s): prune with --drop-cols %s",
				strings.Join(zeroGainFeatures, ","), strings.Join(suggestedDropCols, ",")))
		}
	}

	if len(suggestedDropCols) > 0 {
		cliArgs = append(cliArgs, "--drop-cols", strings.Join(suggestedDropCols, ","))
	}

	if len(suggestions) == 0 {
		suggestions = []string{"curve is healthy; proceed to fine-tune around current parent or publish"}
	}

	report := map[string]any{
		"schema_version":      metricsSchemaVersion,
		"diagnosis":           diagnosis,
		"maximize":            maximize,
		"n_rounds_logged":     len(items),
		"best_round":          bestRound,
		"best_val":            bestVal,
		"final_train":         finalTrain,
		"final_val":           finalVal,
		"overfit_gap":         overfitGap,
		"suggested_mutations": suggestions,
		"suggested_cli_args":  cliArgs,
	}
	if len(secondaryDiagnoses) > 0 {
		report["secondary_diagnoses"] = secondaryDiagnoses
	}
	if *modelPath != "" {
		report["zero_gain_features"] = zeroGainFeatures
		report["top_features"] = topFeatures
	}
	if len(suggestedDropCols) > 0 {
		report["suggested_drop_cols"] = suggestedDropCols
	}
	if *dataPath != "" {
		report["data_warnings"] = dataWarnings
		if len(constantFeatures) > 0 {
			report["constant_features"] = constantFeatures
		}
		if len(idLikeCols) > 0 {
			report["id_like_columns"] = idLikeCols
		}
		if len(categoricalCols) > 0 {
			report["categorical_columns"] = categoricalCols
		}
		if suggestedScalePosWeight > 0 {
			report["suggested_scale_pos_weight"] = suggestedScalePosWeight
		}
	}
	return writeJSON(*outPath, report)
}

func buildDiagnoseCLIArgs(diagnosis string, bestRound int, p *paramsRecord) []string {
	var args []string
	switch diagnosis {
	case "overfitting":
		if bestRound > 0 && (p == nil || p.EarlyStop <= 0) {
			args = append(args, "--early-stop", "20")
		}
		newDepth := 4
		newLambda := 3.0
		if p != nil {
			if p.Depth > 2 {
				newDepth = p.Depth - 1
			} else {
				newDepth = p.Depth
			}
			if p.Lambda >= 1.0 {
				newLambda = p.Lambda * 2.0
			}
		}
		args = append(args, "--depth", strconv.Itoa(newDepth), "--lambda", strconv.FormatFloat(newLambda, 'f', -1, 64))
		if p == nil || p.Subsample >= 1.0 {
			args = append(args, "--subsample", "0.8", "--colsample", "0.8")
		}
	case "underfitting":
		newDepth := 8
		newRounds := 100
		if p != nil {
			newDepth = p.Depth + 2
			if newDepth > 12 {
				newDepth = 12
			}
			newRounds = p.Rounds * 2
			if newRounds < 50 {
				newRounds = 50
			}
		}
		args = append(args, "--depth", strconv.Itoa(newDepth), "--rounds", strconv.Itoa(newRounds))
	case "oscillating":
		newLR := 0.05
		newRounds := 100
		if p != nil {
			if p.LR > 0.01 {
				newLR = p.LR * 0.5
			}
			newRounds = p.Rounds * 2
		}
		args = append(args, "--lr", strconv.FormatFloat(newLR, 'f', -1, 64), "--rounds", strconv.Itoa(newRounds))
	case "high_variance":
		newDepth := 4
		newMCW := 5.0
		if p != nil && p.Depth > 3 {
			newDepth = p.Depth - 2
		}
		if p != nil && p.MinChildWeight >= 1.0 {
			newMCW = p.MinChildWeight * 2.0
		}
		args = append(args, "--depth", strconv.Itoa(newDepth), "--min-child-weight", strconv.FormatFloat(newMCW, 'f', -1, 64))
	}
	return args
}
