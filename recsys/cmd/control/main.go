// 控制面 CLI：推荐生产闭环八段剧本的 shell 入口（演进方案 §十七 RC）。
// Agent 只依赖退出码 + 结构化文件；promote/rollback 请求打印到 stdout，
// 由调用方转发给应用侧 adapter（CLI 不执行任何网络副作用）。
package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"strings"
)

const usage = `control — recsys 推荐生产闭环控制面 CLI

用法:
  control [--error-format text|json] <subcommand> [flags]

  control pipeline   -workspace DIR [-interactions raw.tsv -catalog items.tsv | -synth] [-recall-size 100] [-deck-size 10] [-max-same-tag 3] [-rounds 25] [-out pipeline_report.json]
  control snapshot   -workspace DIR -out snapshot.json -snapshot-id ID -purpose train|eval|release -time-start T -time-end T [-legacy] [-created-at T]
  control split      -events events.jsonl -train-end T -val-start T -test-start T -out-dir DIR
  control eval       -workspace DIR -thresholds th.json [-out evaluation.json] [-recall-k 100] [-ndcg-k 10] [-deck-size 10] [-max-same-tag 3] [-leak-count 0] [-event-count 0]
  control from-deal  -workspace DIR -ledger ledger.jsonl -model-version V -policy-version P -occurred-at T [-feature-schema-hash H] [-candidate-set-id C]
  control append-exposure -ledger L -in exposures.jsonl
  control append-feedback -ledger L -in feedback.jsonl
  control replay     -ledger L -out samples.jsonl [-report replay_report.json] [-window 24h] [-negative impressed_no_feed|none] [-positive-threshold F]
  control monitor    -ledger L -workspace DIR -window-start T -window-end T [-thresholds th.json] [-triggers tr.json] [-out monitor_report.json] [-fired fired.jsonl] [-deck-size 10] [-max-same-tag 3]
  control release    -state release_state.json -action candidate|approve|confirm-promote|observe|retrain|rollback|retire|status [-release-id R] [-evaluation evaluation.json] [-model m.leaves.json] [-run-id R] [-snapshot-id S] [-policy-version P] [-approver A] [-model-version V] [-reason REASON] [-at T]

T = RFC3339（如 2026-08-20T10:00:00Z）。全部输出为结构化文件；
退出码: 0 成功 / 1 用法或 IO / 2 校验或内部错误。
`

// controlJSONOutput 标记是否处于纯 JSON 输出模式（WP-28 / AN8）。
var controlJSONOutput = false

// errUsage 标记用法类错误（退出码 1）。
var errUsage = errors.New("usage")

func usageErr(format string, args ...any) error {
	return fmt.Errorf("%w: %s", errUsage, fmt.Sprintf(format, args...))
}

func exitCode(err error) int {
	if errors.Is(err, errUsage) {
		return 1
	}
	return 2
}

func stripControlGlobalFlags(args []string) []string {
	if strings.EqualFold(strings.TrimSpace(os.Getenv("LEAVES_ERROR_FORMAT")), "json") {
		controlJSONOutput = true
	}
	out := make([]string, 0, len(args))
	for i := 0; i < len(args); i++ {
		a := args[i]
		switch {
		case (a == "--error-format" || a == "-error-format") && i+1 < len(args):
			controlJSONOutput = strings.EqualFold(strings.TrimSpace(args[i+1]), "json")
			i++
		case strings.HasPrefix(a, "--error-format="):
			controlJSONOutput = strings.EqualFold(strings.TrimSpace(strings.TrimPrefix(a, "--error-format=")), "json")
		case strings.HasPrefix(a, "-error-format="):
			controlJSONOutput = strings.EqualFold(strings.TrimSpace(strings.TrimPrefix(a, "-error-format=")), "json")
		case a == "--json" || a == "-json":
			controlJSONOutput = true
		default:
			out = append(out, a)
		}
	}
	return out
}

func main() {
	args := stripControlGlobalFlags(os.Args[1:])
	if len(args) < 1 {
		fmt.Fprint(os.Stderr, usage)
		os.Exit(1)
	}
	sub, rest := args[0], args[1:]
	var err error
	switch sub {
	case "pipeline":
		err = cmdPipeline(rest)
	case "snapshot":
		err = cmdSnapshot(rest)
	case "split":
		err = cmdSplit(rest)
	case "eval":
		err = cmdEval(rest)
	case "from-deal":
		err = cmdFromDeal(rest)
	case "append-exposure":
		err = cmdAppend(rest, "exposure")
	case "append-feedback":
		err = cmdAppend(rest, "feedback")
	case "replay":
		err = cmdReplay(rest)
	case "monitor":
		err = cmdMonitor(rest)
	case "release":
		err = cmdRelease(rest)
	case "help", "-h", "--help":
		fmt.Print(usage)
		return
	default:
		err = usageErr("未知子命令 %q", sub)
	}
	if err != nil {
		code := exitCode(err)
		if controlJSONOutput {
			errCode := "internal"
			if code == 1 {
				errCode = "usage"
			}
			b, _ := json.MarshalIndent(map[string]any{
				"error":      errCode,
				"subcommand": sub,
				"message":    err.Error(),
				"exit_code":  code,
			}, "", "  ")
			fmt.Fprintln(os.Stderr, string(b))
		} else {
			fmt.Fprintf(os.Stderr, "control %s: %v\n", sub, err)
		}
		os.Exit(code)
	}
}
