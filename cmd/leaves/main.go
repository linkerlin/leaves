// Command leaves 是 Agent 友好的训练/评估/预测/发布 CLI（SKILL 驱动，无 MCP）。
//
// 闭环：train(--cv) → 读 metrics.json → 按 SKILL 决策表调参 → 再训 → 收敛 → publish。
// 详见 skills/leaves-autotrain/SKILL.md 与 cli.md。
package main

import (
	"fmt"
	"os"
	"runtime/debug"

	_ "github.com/linkerlin/leaves/v2" // 注册 io 加载器（leaves.json/XGB/LGB）
)

const usage = `leaves — Agent 友好的训练/评估/预测/发布 CLI（SKILL 驱动，无 MCP）

用法:
  leaves [--error-format text|json] <subcommand> [flags]

子命令:
  train    --data PATH --objective NAME [flags]   训练，输出模型 + metrics.json
           --from-run / --na-policy / --auto-encode / --label-col / --drop-cols / --weight-col …
  autotune --data PATH [--budget N] [flags]       有界预算反射式演化自治闭环（sniff→baseline→diagnose→propose/crossover）
  split    --data PATH --out-dir DIR [flags]      零泄漏数据切分（random|stratified|group|time）
  runs     <status|pareto|select-parent|check-dup|crossover|propose> 账本查询、Pareto 选父与演化算子
  diagnose [--metrics PATH] [--model PATH] [--rounds PATH] [--data PATH] 全息 ASI 诊断与变异建议
  eval     --model PATH --data PATH [flags]       评估已存模型（支持 leaves.json/XGB/LGB/ONNX）
  compare  --models M1,M2 --data PATH [flags]     多模型同台评测与配对 Bootstrap 显著性检验
  predict  --model PATH --data PATH --out PATH    批预测 → JSONL/CSV（支持 --id-col / --output / --threshold）
  inspect  --model PATH [--metrics PATH]          模型元数据 → JSON
  sniff    --data PATH [--metrics PATH]           数据画像 → 推荐 objective 与脏数据检测
  explain  --model PATH [--type importance|shap|shap-summary|interaction|dump] 深度可解释性
  publish  --model PATH --out-dir DIR [flags]     本地工件包（--quantize / --export-xgb / --export-onnx）
  lessons  <add|search|list> [flags]              跨任务记忆库（~/.leaves/lessons.jsonl）
  schema   [--command NAME] [--metrics PATH]      CLI 机器可读参数与输出契约
  version                                         版本/构建信息 → JSON

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
	case "autotune", "evolve":
		err = cmdAutotune(args[1:])
	case "split":
		err = cmdSplit(args[1:])
	case "runs":
		err = cmdRuns(args[1:])
	case "diagnose":
		err = cmdDiagnose(args[1:])
	case "eval":
		err = cmdEval(args[1:])
	case "compare":
		err = cmdCompare(args[1:])
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
	case "lessons":
		err = cmdLessons(args[1:])
	case "schema":
		err = cmdSchema(args[1:])
	case "version":
		err = writeJSON("", buildVersionDoc())
	case "-h", "--help", "help":
		fmt.Print(usage)
		return
	default:
		err = errAgent("usage", fmt.Sprintf("未知子命令: %s", args[0]),
			"合法子命令: train|autotune|split|runs|diagnose|eval|compare|predict|inspect|sniff|explain|publish|lessons|schema|version", false)
		os.Exit(writeError(err))
	}
	if err != nil {
		os.Exit(writeError(err))
	}
}

// cliVersionLabel 供 manifest.leaves_cli：tag 或 "(devel)+<短commit>"。
func cliVersionLabel() string {
	doc := buildVersionDoc()
	v, _ := doc["version"].(string)
	if c, ok := doc["commit"].(string); ok && len(c) >= 8 {
		return v + "+" + c[:8]
	}
	return v
}

// buildVersionDoc 从 go build 信息取真实版本：
//   - go install pkg@vX.Y.Z → Main.Version = "vX.Y.Z"
//   - 仓库内 go build/run → "(devel)" + vcs.revision
func buildVersionDoc() map[string]any {
	doc := map[string]any{"go": "?", "version": "unknown"}
	if bi, ok := debug.ReadBuildInfo(); ok {
		doc["go"] = bi.GoVersion
		doc["version"] = bi.Main.Version
		for _, s := range bi.Settings {
			if s.Key == "vcs.revision" && s.Value != "" {
				doc["commit"] = s.Value
			}
		}
	}
	return doc
}
