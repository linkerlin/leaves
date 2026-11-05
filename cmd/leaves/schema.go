package main

import (
	"flag"
	"fmt"
	"strings"
)

func cmdSchema(args []string) error {
	fs := flag.NewFlagSet("schema", flag.ContinueOnError)
	fs.Usage = func() {
		fmt.Fprintln(fs.Output(), "用法: leaves schema [--command NAME] [--metrics PATH]")
	}
	cmdName := fs.String("command", "", "仅输出指定子命令 schema（空=输出全量）")
	subcmdAlias := fs.String("subcommand", "", "--command 别名")
	metricsPath := fs.String("metrics", "", "输出 JSON 路径（空=stdout）")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *cmdName == "" && *subcmdAlias != "" {
		*cmdName = *subcmdAlias
	}

	commands := map[string]any{
		"sniff": map[string]any{
			"description":   "数据画像、特征质量扫描与目标函数自动推荐",
			"required_args": []string{"--data"},
			"optional_args": []string{"--label-col", "--drop-cols", "--na-policy", "--metrics"},
			"output_fields": []string{"schema_version", "file", "format", "n_rows", "n_features", "has_label", "has_qid", "label", "suggested_objective", "suggested_metric", "data_quality"},
		},
		"split": map[string]any{
			"description":   "零泄漏确定性数据切分（random|stratified|group|time，支持 --na-policy nan 与标签漂移审计）",
			"required_args": []string{"--data", "--out-dir"},
			"optional_args": []string{"--ratios", "--strategy", "--label-col", "--drop-cols", "--group-col", "--time-col", "--seed", "--na-policy", "--metrics"},
			"output_fields": []string{"schema_version", "strategy", "ratios", "total_rows", "skipped_rows", "missing_cells", "leakage_ok", "train_path", "val_path", "test_path", "train_rows", "val_rows", "test_rows", "label_means", "max_label_drift"},
		},
		"train": map[string]any{
			"description":   "训练或交叉验证 GBDT/DART/GBLinear 模型并记录账本",
			"required_args": []string{"--data", "--objective (or --from-run / --resume)"},
			"optional_args": []string{
				"--eval-metric", "--num-class", "--num-target", "--rounds", "--depth", "--max-leaves",
				"--lr", "--lambda", "--min-child-weight", "--gamma", "--max-bin", "--subsample", "--colsample",
				"--tree-method", "--ndcg-k", "--cv", "--val", "--early-stop", "--seed", "--out-model", "--out-final",
				"--metrics", "--runs", "--tag", "--emit-rounds", "--save-best", "--from-run", "--na-policy", "--auto-encode",
				"--strict-flags", "--label-col", "--drop-cols", "--weight-col", "--scale-pos-weight",
				"--tweedie-variance-power", "--multi-output-tree", "--num-parallel-tree",
				"--booster", "--rate-drop", "--skip-drop",
				"--monotone", "--checkpoint", "--checkpoint-every", "--resume", "--batch-rows", "--lr-decay", "--dry-run",
			},
			"output_fields": []string{"schema_version", "objective", "metric", "value", "maximize", "n_rows", "n_features", "cv_folds", "cv_mean", "cv_std", "fold_metrics", "cv_best_rounds", "cv_best_round_mean", "best_round", "stopped_round", "model_round", "n_trees", "elapsed_ms", "binary_diagnostics", "regression_diagnostics", "multiclass_diagnostics", "ranking_diagnostics", "params"},
		},
		"autotune": map[string]any{
			"description":   "有界预算反射式演化自治闭环（自动串联 sniff -> lessons 召回 -> baseline -> diagnose -> propose/crossover -> 定稿 -> lessons 沉淀）",
			"required_args": []string{"--data"},
			"optional_args": []string{"--val", "--objective", "--eval-metric", "--num-class", "--num-target", "--cv", "--budget", "--rounds", "--early-stop", "--label-col", "--drop-cols", "--weight-col", "--na-policy", "--auto-encode", "--out-dir", "--out-model", "--metrics", "--task", "--record-lesson", "--seed"},
			"output_fields": []string{"schema_version", "task", "objective", "metric", "maximize", "budget", "runs_executed", "baseline_tag", "baseline_value", "best_tag", "best_value", "improvement_pct", "best_model", "runs_ledger", "best_params", "auto_adaptations", "prior_lessons", "distilled_lesson", "lineage"},
		},
		"runs": map[string]any{
			"description":   "实验账本查询、折级/成本 Pareto 前沿计算、加权选父、交叉重组、定向变异提议与超参去重",
			"subcommands":   []string{"status", "pareto", "select-parent", "check-dup", "crossover", "propose"},
			"required_args": []string{"--runs"},
			"optional_args": []string{"--strategy", "--params", "--parent-a", "--parent-b", "--parent", "--diagnose", "--seed", "--rel-tol", "--patience", "--metrics"},
		},
		"diagnose": map[string]any{
			"description":   "全息 ASI 诊断引擎（学习曲线 + 模型零增益特征 + 数据质量 + 可执行变异 CLI 建议）",
			"required_args": []string{"--model or --rounds-log (--rounds) or --metrics or --data"},
			"optional_args": []string{"--model", "--rounds-log", "--rounds", "--metrics", "--data", "--label-col", "--drop-cols", "--out"},
			"output_fields": []string{"schema_version", "diagnosis", "best_round", "best_val", "final_train", "final_val", "overfit_gap", "zero_gain_features", "top_features", "suggested_drop_cols", "suggested_mutations", "suggested_cli_args"},
		},
		"eval": map[string]any{
			"description":   "在独立数据集上评估模型（支持 leaves.json/XGB/LGB/sklearn/ONNX，自动输出分类/回归/排序扩展诊断）",
			"required_args": []string{"--model", "--data"},
			"optional_args": []string{"--eval-metric", "--objective", "--num-target", "--label-col", "--drop-cols", "--weight-col", "--na-policy", "--auto-encode", "--metrics"},
			"output_fields": []string{"schema_version", "objective", "metric", "value", "maximize", "n_rows", "n_features", "binary_diagnostics", "regression_diagnostics", "multiclass_diagnostics", "ranking_diagnostics"},
		},
		"compare": map[string]any{
			"description":   "多模型同台评测与配对 Bootstrap 显著性检验（支持跨格式对比）",
			"required_args": []string{"--models", "--data"},
			"optional_args": []string{"--eval-metric", "--objective", "--bootstrap", "--seed", "--num-target", "--label-col", "--drop-cols", "--na-policy", "--auto-encode", "--metrics"},
			"output_fields": []string{"schema_version", "baseline_model", "winner_model", "winner_index", "models"},
		},
		"predict": map[string]any{
			"description":   "批量模型推理（支持全目标 Link 逆变换、value/margin/leaf/contrib 输出、--id-col 透传与 --metrics 统计摘要）",
			"required_args": []string{"--model", "--data", "--out"},
			"optional_args": []string{"--format", "--output", "--id-col", "--label-col", "--drop-cols", "--objective", "--na-policy", "--auto-encode", "--threshold", "--metrics"},
			"output_fields": []string{"schema_version", "model", "data", "out", "objective", "output_kind", "format", "n_rows", "n_features", "pred_min", "pred_max", "pred_mean", "pred_std", "positive_rate", "elapsed_ms"},
		},
		"inspect": map[string]any{
			"description":   "查看模型深度结构画像（支持 leaves.json/.quant.json/XGB/LGB/sklearn/ONNX，含树深度、叶数分布与未使用特征列表）",
			"required_args": []string{"--model"},
			"optional_args": []string{"--metrics"},
			"output_fields": []string{"schema_version", "file", "format", "objective", "kind", "num_features", "n_output_groups", "n_trees", "n_boost_rounds", "base_score", "max_depth_observed", "total_nodes", "total_leaves", "mean_leaves_per_tree", "used_features_count", "unused_features_count", "unused_features"},
		},
		"explain": map[string]any{
			"description":   "多维特征重要性（gain|weight|cover）、Tree SHAP（逐行与全局 summary）、二阶交互 SHAP（interaction）与树结构导出（dump）",
			"required_args": []string{"--model"},
			"optional_args": []string{"--type", "--importance-type", "--summary", "--dump-format", "--data", "--max-rows", "--label-col", "--drop-cols", "--na-policy", "--auto-encode", "--metrics"},
		},
		"publish": map[string]any{
			"description":   "打包发布产物目录（model.leaves.json、可选 XGB/ONNX 导出、int8 量化、manifest.json 与复现脚本）",
			"required_args": []string{"--model", "--out-dir"},
			"optional_args": []string{"--version", "--quantize", "--export-xgb", "--export-onnx", "--metrics", "--data", "--label-col", "--drop-cols", "--na-policy", "--emit-repro-script", "--print-repro"},
		},
		"lessons": map[string]any{
			"description":   "跨任务经验记忆库追加、关键词检索与列表（支持 JSONL stdout 与 --metrics 结构化 JSON）",
			"subcommands":   []string{"add", "search", "list"},
			"optional_args": []string{"--task", "--lesson", "--evidence", "--tag", "--query", "--limit", "--metrics"},
			"output_fields": []string{"schema_version", "action", "count", "lessons"},
		},
	}

	target := strings.ToLower(strings.TrimSpace(*cmdName))
	if target != "" {
		c, ok := commands[target]
		if !ok {
			return errUsage(fmt.Sprintf("未知子命令 %q", target))
		}
		return writeJSON(*metricsPath, map[string]any{
			"schema_version": metricsSchemaVersion,
			"command":        target,
			"spec":           c,
		})
	}

	return writeJSON(*metricsPath, map[string]any{
		"schema_version": metricsSchemaVersion,
		"cli":            "leaves",
		"global_flags":   []string{"--error-format text|json"},
		"error_codes":    []string{"usage", "data_load", "non_numeric", "missing_value", "missing_label", "feature_mismatch", "objective_mismatch", "model_load", "cv_conflict", "internal"},
		"commands":       commands,
	})
}
