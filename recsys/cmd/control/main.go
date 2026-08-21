// 控制面 CLI：推荐生产闭环八段剧本的 shell 入口（演进方案 §十七 RC）。
// Agent 只依赖退出码 + 结构化文件；promote/rollback 请求打印到 stdout，
// 由调用方转发给应用侧 adapter（CLI 不执行任何网络副作用）。
package main

import (
	"errors"
	"fmt"
	"os"
)

const usage = `control — recsys 推荐生产闭环控制面 CLI

用法:
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

func main() {
	if len(os.Args) < 2 {
		fmt.Fprint(os.Stderr, usage)
		os.Exit(1)
	}
	sub, args := os.Args[1], os.Args[2:]
	var err error
	switch sub {
	case "snapshot":
		err = cmdSnapshot(args)
	case "split":
		err = cmdSplit(args)
	case "eval":
		err = cmdEval(args)
	case "from-deal":
		err = cmdFromDeal(args)
	case "append-exposure":
		err = cmdAppend(args, "exposure")
	case "append-feedback":
		err = cmdAppend(args, "feedback")
	case "replay":
		err = cmdReplay(args)
	case "monitor":
		err = cmdMonitor(args)
	case "release":
		err = cmdRelease(args)
	case "help", "-h", "--help":
		fmt.Print(usage)
		return
	default:
		fmt.Fprintf(os.Stderr, "control: 未知子命令 %q\n\n%s", sub, usage)
		os.Exit(1)
	}
	if err != nil {
		fmt.Fprintf(os.Stderr, "control %s: %v\n", sub, err)
		os.Exit(exitCode(err))
	}
}
