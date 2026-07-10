// Command leaves 是 Agent 友好的训练/评估/预测/发布 CLI（SKILL 驱动，无 MCP）。
//
// 闭环：train(--cv) → 读 metrics.json → 按 SKILL 决策表调参 → 再训 → 收敛 → publish。
// 详见 skills/leaves-autotrain/SKILL.md 与 cli.md。
package main

import (
	"errors"
	"fmt"
	"os"

	_ "github.com/linkerlin/leaves" // 注册 io 加载器（leaves.json/XGB/LGB）
)

const usage = `leaves — Agent 友好的训练/评估/预测/发布 CLI（SKILL 驱动，无 MCP）

用法:
  leaves train    --data PATH --objective NAME [flags]   训练，输出模型 + metrics.json
  leaves eval     --model PATH --data PATH [flags]        评估已存模型
  leaves predict  --model PATH --data PATH --out PATH     批预测 → JSONL
  leaves inspect  --model PATH [--metrics PATH]           模型元数据 → JSON
  leaves sniff    --data PATH [--metrics PATH]            数据画像 → 推荐 objective
  leaves explain  --model PATH [--type importance|shap]   特征重要性 / SHAP
  leaves publish  --model PATH --out-dir DIR [flags]      打成本地工件包 + manifest

闭环：train(--cv) → 读 metrics.json → 按 SKILL 决策表调参 → 再训 → 收敛 → publish
详见 skills/leaves-autotrain/cli.md（flag 全表 + metrics.json schema）
`

type usageError string

func (e usageError) Error() string { return string(e) }

func errUsage(s string) error { return usageError(s) }

func main() {
	if len(os.Args) < 2 {
		fmt.Fprint(os.Stderr, usage)
		os.Exit(1)
	}
	var err error
	switch os.Args[1] {
	case "train":
		err = cmdTrain(os.Args[2:])
	case "eval":
		err = cmdEval(os.Args[2:])
	case "predict":
		err = cmdPredict(os.Args[2:])
	case "inspect":
		err = cmdInspect(os.Args[2:])
	case "sniff":
		err = cmdSniff(os.Args[2:])
	case "explain":
		err = cmdExplain(os.Args[2:])
	case "publish":
		err = cmdPublish(os.Args[2:])
	case "-h", "--help", "help":
		fmt.Print(usage)
		return
	default:
		fmt.Fprintf(os.Stderr, "未知子命令: %s\n\n%s", os.Args[1], usage)
		os.Exit(1)
	}
	if err != nil {
		var ue usageError
		if errors.As(err, &ue) {
			fmt.Fprintf(os.Stderr, "leaves: 参数错误: %v\n", err)
			os.Exit(1)
		}
		fmt.Fprintf(os.Stderr, "leaves: %v\n", err)
		os.Exit(2)
	}
}
