package main

import (
	"flag"
	"fmt"
	"strings"

	"github.com/linkerlin/leaves/v2/data"
)

func cmdSplit(args []string) error {
	fs := flag.NewFlagSet("split", flag.ContinueOnError)
	fs.Usage = func() {
		fmt.Fprintln(fs.Output(), "用法: leaves split --data PATH --out-dir DIR [--ratios 0.7,0.15,0.15] [--strategy auto|random|stratified|group|time] [flags]")
	}
	dataPath := fs.String("data", "", "原始数据路径（必需）")
	outDir := fs.String("out-dir", "", "切分产物输出目录（必需）")
	ratiosStr := fs.String("ratios", "0.7,0.15,0.15", "train,val[,test] 切分比例")
	strategy := fs.String("strategy", "auto", "切分策略：auto|random|stratified|group|time")
	labelCol := fs.String("label-col", "", "显式指定标签列名或 0-based 索引（用于 stratified 切分）")
	dropCols := fs.String("drop-cols", "", "逗号分隔需丢弃的列名或索引")
	groupCol := fs.String("group-col", "", "分组列名或 0-based 索引（用于 group 零泄漏切分）")
	timeCol := fs.String("time-col", "", "时间列名或 0-based 索引（用于 time 升序切分）")
	seed := fs.Int64("seed", 42, "确定性随机种子")
	naPolicy := fs.String("na-policy", "error", "缺失值策略：error|skip-row|nan（nan 原样保留缺失特征行，WP-47）")
	metricsPath := fs.String("metrics", "", "额外输出 split 报告 JSON 路径（空=stdout；out-dir/split.json 始终写出）")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *dataPath == "" || *outDir == "" {
		return errUsage("--data 与 --out-dir 必需")
	}
	ratios, err := data.ParseSplitRatios(*ratiosStr)
	if err != nil {
		return errUsage(fmt.Sprintf("--ratios: %v", err))
	}
	var dropList []string
	if strings.TrimSpace(*dropCols) != "" {
		dropList = []string{*dropCols}
	}
	rep, err := data.SplitDataset(data.SplitOptions{
		DataPath: *dataPath,
		OutDir:   *outDir,
		Ratios:   ratios,
		Strategy: *strategy,
		LabelCol: *labelCol,
		DropCols: dropList,
		GroupCol: *groupCol,
		TimeCol:  *timeCol,
		Seed:     *seed,
		NAPolicy: *naPolicy,
	})
	if err != nil {
		return errAgentWrap("data_load", fmt.Sprintf("split %s: %v", *dataPath, err),
			"检查 --strategy、--label-col、--group-col、--time-col 与数据格式", false, err)
	}
	return writeJSON(*metricsPath, rep)
}
