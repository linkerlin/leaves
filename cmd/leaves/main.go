// Command leaves 是 Agent 友好的训练/评估/预测/发布 CLI（SKILL 驱动，无 MCP）。
//
// 闭环：train(--cv) → 读 metrics.json → 按 SKILL 决策表调参 → 再训 → 收敛 → publish。
// 详见 skills/leaves-autotrain/SKILL.md 与 cli.md。
package main

import (
	"fmt"
	"os"

	_ "github.com/linkerlin/leaves" // 注册 io 加载器（leaves.json/XGB/LGB）
)

const usage = `leaves — Agent 友好的训练/评估/预测/发布 CLI（SKILL 驱动，无 MCP）

用法:
  leaves [--error-format text|json] <subcommand> [flags]

子命令:
  train    --data PATH --objective NAME [flags]   训练，输出模型 + metrics.json
           --from-run / --na-policy / --save-best …
  eval     --model PATH --data PATH [flags]        评估已存模型
  predict  --model PATH --data PATH --out PATH     批预测 → JSONL
  inspect  --model PATH [--metrics PATH]           模型元数据 → JSON
  sniff    --data PATH [--metrics PATH]            数据画像 → 推荐 objective
  explain  --model PATH [--type importance|shap]   特征重要性 / SHAP
  publish  --model PATH --out-dir DIR [flags]      本地工件包（--emit-repro-script）

全局:
  --error-format text|json   错误输出格式（默认 text；json 供 Agent 解析）
  环境变量 LEAVES_ERROR_FORMAT=json 等价

闭环：sniff → train(--cv/--runs) → 读 metrics → 调参 → 收敛 → publish
详见 skills/leaves-autotrain/cli.md
`

type usageError string

func (e usageError) Error() string { return string(e) }

func errUsage(s string) error { return usageError(s) }

func main() {
	args := stripGlobalFlags(os.Args[1:])
	if len(args) < 1 {
		fmt.Fprint(os.Stderr, usage)
		os.Exit(1)
	}
	var err error
	switch args[0] {
	case "train":
		err = cmdTrain(args[1:])
	case "eval":
		err = cmdEval(args[1:])
	case "predict":
		err = cmdPredict(args[1:])
	case "inspect":
		err = cmdInspect(args[1:])
	case "sniff":
		err = cmdSniff(args[1:])
	case "explain":
		err = cmdExplain(args[1:])
	case "publish":
		err = cmdPublish(args[1:])
	case "-h", "--help", "help":
		fmt.Print(usage)
		return
	default:
		err = errAgent("usage", fmt.Sprintf("未知子命令: %s", args[0]),
			"合法子命令: train|eval|predict|inspect|sniff|explain|publish", false)
		os.Exit(writeError(err))
	}
	if err != nil {
		os.Exit(writeError(err))
	}
}
