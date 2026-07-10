package main

import (
	"flag"
	"fmt"

	leavesio "github.com/linkerlin/leaves/io"
	"github.com/linkerlin/leaves/metrics"
)

func cmdEval(args []string) error {
	fs := flag.NewFlagSet("eval", flag.ContinueOnError)
	fs.Usage = func() { fmt.Fprintln(fs.Output(), "用法: leaves eval --model PATH --data PATH [flags]") }
	modelPath := fs.String("model", "", "模型路径（必需）")
	dataPath := fs.String("data", "", "数据路径（必需）")
	evalMetric := fs.String("eval-metric", "", "评估指标（默认 rmse）")
	objective := fs.String("objective", "", "目标函数（margin→pred 变换；可从 metric 推断）")
	metricsPath := fs.String("metrics", "", "输出 metrics.json（空=stdout）")
	naPolicy := fs.String("na-policy", "error", "缺失值策略：error|skip-row")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *modelPath == "" || *dataPath == "" {
		return errUsage("--model 与 --data 必需")
	}
	metric := *evalMetric
	obj := *objective
	if obj == "" {
		// 从模型元数据自动推断 objective（排名默认 ndcg@10，二分类默认 logloss，等）
		if _, irObj, err := leavesio.ParseLeavesJSONFile(*modelPath); err == nil {
			obj = irObj
		}
	}
	if metric == "" {
		metric = defaultMetric(obj)
	}

	m, err := loadEnsemble(*modelPath)
	if err != nil {
		return fmt.Errorf("load model: %w", err)
	}
	defer m.Close()

	dm, err := loadMatrix(*dataPath, *naPolicy)
	if err != nil {
		return err
	}
	vals, err := denseVals(dm)
	if err != nil {
		return err
	}

	nc := m.NOutputGroups()
	if nc < 1 {
		nc = 1
	}
	preds := make([]float64, dm.NumRow()*nc)
	if err := m.PredictDense(vals, dm.NumRow(), dm.NumCol(), preds, 0, 0); err != nil {
		return fmt.Errorf("predict: %w", err)
	}

	groups := groupsOf(dm)
	mt, err := metrics.Resolve(metric, metrics.Options{NumClass: nc, Groups: groups})
	if err != nil {
		return fmt.Errorf("resolve metric %q: %w", metric, err)
	}
	metricPreds := metricInputs(metric, obj, preds, nc)
	score, err := metrics.Evaluate(mt, dm.Labels(), metricPreds, groups)
	if err != nil {
		return fmt.Errorf("evaluate: %w", err)
	}

	return writeMetrics(*metricsPath, metricsDoc{
		Objective: obj,
		Metric:    metric,
		Value:     score,
		Maximize:  mt.HigherIsBetter(),
		NRows:     dm.NumRow(),
	})
}
