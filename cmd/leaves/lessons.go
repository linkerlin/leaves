package main

// lessons.go — 跨任务记忆库 CLI（LES-2：单目录 lessons.md 升级为可检索存储）。
//
// 存储：~/.leaves/lessons.jsonl（追加写；LEAVES_LESSONS_PATH 可覆盖，测试/多环境用）。
// 定位：只做存储与检索管道；「写什么、何时读」的策略仍在 SKILL（§4.6）。
// Agent 检索：leaves lessons search --query "..."；沉淀：leaves lessons add ...。

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"
)

type lessonRecord struct {
	TS       string   `json:"ts"`
	Task     string   `json:"task"`
	Lesson   string   `json:"lesson"`
	Evidence string   `json:"evidence,omitempty"`
	Tags     []string `json:"tags,omitempty"`
}

func lessonsPath() string {
	if p := os.Getenv("LEAVES_LESSONS_PATH"); p != "" {
		return p
	}
	home, err := os.UserHomeDir()
	if err != nil || home == "" {
		return "lessons.jsonl"
	}
	return filepath.Join(home, ".leaves", "lessons.jsonl")
}

func cmdLessons(args []string) error {
	if len(args) < 1 {
		return errUsage("用法: leaves lessons <add|search|list> [flags]")
	}
	switch args[0] {
	case "add":
		return lessonsAdd(args[1:])
	case "search":
		return lessonsSearch(args[1:])
	case "list":
		return lessonsList(args[1:])
	default:
		return errUsage(fmt.Sprintf("未知 lessons 子命令: %s（合法: add|search|list）", args[0]))
	}
}

func lessonsAdd(args []string) error {
	fs := flag.NewFlagSet("lessons add", flag.ContinueOnError)
	fs.Usage = func() {
		fmt.Fprintln(fs.Output(), "用法: leaves lessons add --task X --lesson \"...\" [--evidence \"...\"] [--tag a,b]")
	}
	task := fs.String("task", "", "任务签名（必需）")
	lesson := fs.String("lesson", "", "教训一句话（必需）")
	evidence := fs.String("evidence", "", "证据（账本 tag / 数字 / 错误码）")
	tags := fs.String("tag", "", "逗号分隔标签")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *task == "" || *lesson == "" {
		return errUsage("--task 与 --lesson 必需")
	}
	rec := lessonRecord{
		TS:       time.Now().UTC().Format(time.RFC3339),
		Task:     *task,
		Lesson:   *lesson,
		Evidence: *evidence,
	}
	if t := strings.TrimSpace(*tags); t != "" {
		for _, x := range strings.Split(t, ",") {
			if x = strings.TrimSpace(x); x != "" {
				rec.Tags = append(rec.Tags, x)
			}
		}
	}
	path := lessonsPath()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return errAgentWrap("data_load", fmt.Sprintf("lessons: mkdir: %v", err), "检查目录权限", false, err)
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return errAgentWrap("data_load", fmt.Sprintf("lessons: open: %v", err), "检查文件权限（LEAVES_LESSONS_PATH 可改路径）", false, err)
	}
	defer f.Close()
	line, err := json.Marshal(rec)
	if err != nil {
		return err
	}
	if _, err := f.Write(append(line, '\n')); err != nil {
		return err
	}
	return writeStdoutLine(rec)
}

type lessonHit struct {
	lessonRecord
	Hits int `json:"hits"`
}

func lessonsSearch(args []string) error {
	fs := flag.NewFlagSet("lessons search", flag.ContinueOnError)
	fs.Usage = func() { fmt.Fprintln(fs.Output(), "用法: leaves lessons search --query \"词...\" [--limit N]") }
	query := fs.String("query", "", "检索词（空格分词，大小写不敏感）")
	limit := fs.Int("limit", 10, "最多输出条数（0=不限）")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if strings.TrimSpace(*query) == "" {
		return errUsage("--query 必需")
	}
	recs, err := readLessons()
	if err != nil {
		return err
	}
	tokens := strings.Fields(strings.ToLower(*query))
	var hits []lessonHit
	for _, r := range recs {
		hay := strings.ToLower(r.Task + " " + r.Lesson + " " + r.Evidence + " " + strings.Join(r.Tags, " "))
		n := 0
		for _, tk := range tokens {
			if strings.Contains(hay, tk) {
				n++
			}
		}
		if n > 0 {
			hits = append(hits, lessonHit{lessonRecord: r, Hits: n})
		}
	}
	// 命中数降序（同分保持时间序的稳定：插入序）
	for i := 0; i < len(hits); i++ {
		for j := i + 1; j < len(hits); j++ {
			if hits[j].Hits > hits[i].Hits {
				hits[i], hits[j] = hits[j], hits[i]
			}
		}
	}
	if *limit > 0 && len(hits) > *limit {
		hits = hits[:*limit]
	}
	for _, h := range hits {
		if err := writeStdoutLine(h); err != nil {
			return err
		}
	}
	return nil
}

func lessonsList(args []string) error {
	fs := flag.NewFlagSet("lessons list", flag.ContinueOnError)
	fs.Usage = func() { fmt.Fprintln(fs.Output(), "用法: leaves lessons list [--task X]") }
	task := fs.String("task", "", "按任务签名过滤（子串，大小写不敏感）")
	if err := fs.Parse(args); err != nil {
		return err
	}
	recs, err := readLessons()
	if err != nil {
		return err
	}
	for _, r := range recs {
		if *task != "" && !strings.Contains(strings.ToLower(r.Task), strings.ToLower(*task)) {
			continue
		}
		if err := writeStdoutLine(r); err != nil {
			return err
		}
	}
	return nil
}

// writeStdoutLine 单行 JSON 输出（JSONL 语义，Agent 逐行解析）。
func writeStdoutLine(v any) error {
	b, err := json.Marshal(v)
	if err != nil {
		return err
	}
	b = append(b, '\n')
	_, err = os.Stdout.Write(b)
	return err
}

func readLessons() ([]lessonRecord, error) {
	path := lessonsPath()
	f, err := os.Open(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil // 空记忆库合法
		}
		return nil, errAgentWrap("data_load", fmt.Sprintf("lessons: open: %v", err), "检查文件权限", false, err)
	}
	defer f.Close()
	var out []lessonRecord
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
	line := 0
	for sc.Scan() {
		line++
		s := strings.TrimSpace(sc.Text())
		if s == "" || strings.HasPrefix(s, "#") {
			continue
		}
		var r lessonRecord
		if err := json.Unmarshal([]byte(s), &r); err != nil {
			return nil, errAgentWrap("data_load",
				fmt.Sprintf("lessons: 第 %d 行非 JSONL（手工编辑破坏？）: %v", line, err),
				fmt.Sprintf("修复或删除 %s 中该行", path), false, err)
		}
		if r.Task == "" || r.Lesson == "" {
			return nil, errAgentWrap("data_load",
				fmt.Sprintf("lessons: 第 %d 行缺 task/lesson", line),
				fmt.Sprintf("修复或删除 %s 中该行", path), false, nil)
		}
		out = append(out, r)
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}
	return out, nil
}
