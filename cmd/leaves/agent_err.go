package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"strings"
)

// errorFormat 全局错误输出格式：text（默认）| json（Agent 友好）。
// 由 main 解析 --error-format 或环境变量 LEAVES_ERROR_FORMAT 设置。
var errorFormat = "text"

// agentError 结构化错误：Agent 读 code/hint，不必解析人类散文。
type agentError struct {
	Code      string `json:"error"`
	Message   string `json:"message"`
	Hint      string `json:"hint,omitempty"`
	Retryable bool   `json:"retryable"`
	cause     error
}

func (e *agentError) Error() string {
	if e == nil {
		return ""
	}
	if e.Message != "" {
		return e.Message
	}
	return e.Code
}

func (e *agentError) Unwrap() error { return e.cause }

func errAgent(code, message, hint string, retryable bool) error {
	return &agentError{Code: code, Message: message, Hint: hint, Retryable: retryable}
}

func errAgentWrap(code, message, hint string, retryable bool, cause error) error {
	return &agentError{Code: code, Message: message, Hint: hint, Retryable: retryable, cause: cause}
}

// classifyError 将任意 error 归一为 agentError（尽量可操作）。
func classifyError(err error) *agentError {
	if err == nil {
		return nil
	}
	var ae *agentError
	if errors.As(err, &ae) {
		return ae
	}
	var ue usageError
	if errors.As(err, &ue) {
		return &agentError{
			Code:      "usage",
			Message:   string(ue),
			Hint:      "检查子命令必需 flag；详见 leaves --help 或 skills/leaves-autotrain/cli.md",
			Retryable: false,
		}
	}
	msg := err.Error()
	low := strings.ToLower(msg)
	switch {
	case strings.Contains(low, "no such file") || strings.Contains(low, "cannot find") ||
		strings.Contains(low, "the system cannot find"):
		return &agentError{Code: "data_load", Message: msg, Hint: "检查 --data/--model 路径是否存在", Retryable: false, cause: err}
	case strings.Contains(low, "parse") && (strings.Contains(low, "csv") || strings.Contains(low, "data")):
		return &agentError{Code: "data_load", Message: msg, Hint: "确认数值 CSV/LIBSVM/ranking TSV；类别列需预编码；空单元格会失败", Retryable: false, cause: err}
	case strings.Contains(low, "non-numeric") || strings.Contains(low, "strconv") || strings.Contains(low, "invalid syntax"):
		return &agentError{Code: "non_numeric", Message: msg, Hint: "所有特征与标签须为数值；先做 one-hot/label encoding", Retryable: false, cause: err}
	case strings.Contains(low, "missing") || strings.Contains(low, "empty cell") || strings.Contains(low, "nan"):
		return &agentError{Code: "missing_value", Message: msg, Hint: "CSV 不允许空单元格；先填补或删行", Retryable: false, cause: err}
	case strings.Contains(low, "num-class") || strings.Contains(low, "num_class") || strings.Contains(low, "numclass"):
		return &agentError{Code: "objective_mismatch", Message: msg, Hint: "multi:* 目标需要 --num-class >= 2（可用 sniff 的 label.n_unique）", Retryable: false, cause: err}
	case strings.Contains(low, "load model") || strings.Contains(low, "parse model") || strings.Contains(low, "leaves.json"):
		return &agentError{Code: "model_load", Message: msg, Hint: "确认 --model 为 leaves.json 或支持的 XGB/LGB 路径", Retryable: false, cause: err}
	default:
		return &agentError{Code: "internal", Message: msg, Hint: "见 stderr 全文；可降 rounds/depth 重试或查数据", Retryable: true, cause: err}
	}
}

// writeError 按 errorFormat 输出错误；返回进程退出码。
// 退出码：1=参数/用法/数据/模型加载；2=训练/评估内部错误。
func writeError(err error) int {
	ae := classifyError(err)
	code := 2
	switch ae.Code {
	case "usage", "data_load", "non_numeric", "missing_value", "objective_mismatch", "model_load":
		code = 1
	}
	if errorFormat == "json" {
		b, _ := json.MarshalIndent(map[string]any{
			"error":     ae.Code,
			"message":   ae.Message,
			"hint":      ae.Hint,
			"retryable": ae.Retryable,
			"exit_code": code,
		}, "", "  ")
		fmt.Fprintln(os.Stderr, string(b))
		return code
	}
	if ae.Code == "usage" {
		fmt.Fprintf(os.Stderr, "leaves: 参数错误: %s\n", ae.Message)
	} else {
		fmt.Fprintf(os.Stderr, "leaves: %s\n", ae.Message)
	}
	if ae.Hint != "" {
		fmt.Fprintf(os.Stderr, "hint: %s\n", ae.Hint)
	}
	return code
}

// stripGlobalFlags 解析全局 flag，返回剩余 argv（含 subcommand）。
// 支持：leaves --error-format=json train ...  与  leaves train ...（环境变量兜底）
func stripGlobalFlags(args []string) []string {
	if v := os.Getenv("LEAVES_ERROR_FORMAT"); v != "" {
		errorFormat = strings.ToLower(strings.TrimSpace(v))
	}
	out := make([]string, 0, len(args))
	for i := 0; i < len(args); i++ {
		a := args[i]
		switch {
		case a == "--error-format" && i+1 < len(args):
			errorFormat = strings.ToLower(args[i+1])
			i++
		case strings.HasPrefix(a, "--error-format="):
			errorFormat = strings.ToLower(strings.TrimPrefix(a, "--error-format="))
		default:
			out = append(out, a)
		}
	}
	if errorFormat != "json" && errorFormat != "text" {
		errorFormat = "text"
	}
	return out
}
