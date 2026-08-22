package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func lessonsTestPath(t *testing.T) string {
	t.Helper()
	t.Setenv("LEAVES_LESSONS_PATH", filepath.Join(t.TempDir(), "lessons.jsonl"))
	return os.Getenv("LEAVES_LESSONS_PATH")
}

func TestLessonsAddSearchRoundtrip(t *testing.T) {
	lessonsTestPath(t)

	if err := cmdLessons([]string{"add", "--task", "ml-ctr-v3", "--lesson", "小数据别上 subsample", "--evidence", "0.312->0.341", "--tag", "small-data,subsample"}); err != nil {
		t.Fatal(err)
	}
	if err := cmdLessons([]string{"add", "--task", "churn-reg", "--lesson", "max_bin=64 抗计数特征噪声", "--evidence", "cv_std 0.09->0.05", "--tag", "max_bin"}); err != nil {
		t.Fatal(err)
	}
	if err := cmdLessons([]string{"add", "--task", "ml-ctr-v4", "--lesson", "subsample 小数据失效（跨任务复验）", "--evidence", "v4 复现 v3 教训", "--tag", "small-data"}); err != nil {
		t.Fatal(err)
	}

	// search：两词命中排序（2 命中 > 1 命中）
	hits := searchTestLessons(t, "subsample 小数据")
	if len(hits) == 0 {
		t.Fatal("no hits")
	}
	recs := readTestLessons(t)
	if len(recs) != 3 {
		t.Fatalf("want 3 records, got %d", len(recs))
	}
	if hits[0].Hits < 2 {
		t.Fatalf("top hit should match both tokens: %+v", hits[0])
	}
	if !strings.Contains(hits[0].Lesson, "subsample") {
		t.Fatalf("top hit unexpected: %+v", hits[0])
	}

	// list --task 过滤
	filtered := listTestLessons(t, "churn")
	if len(filtered) != 1 || filtered[0].Task != "churn-reg" {
		t.Fatalf("task filter: %+v", filtered)
	}

	// 空库 search 合法（无输出、无错误）
	t.Setenv("LEAVES_LESSONS_PATH", filepath.Join(t.TempDir(), "empty.jsonl"))
	if hits := searchTestLessons(t, "x"); len(hits) != 0 {
		t.Fatalf("empty store should return no hits, got %d", len(hits))
	}
}

func readTestLessons(t *testing.T) []lessonRecord {
	t.Helper()
	recs, err := readLessons()
	if err != nil {
		t.Fatal(err)
	}
	return recs
}

func searchTestLessons(t *testing.T, query string) []lessonHit {
	t.Helper()
	// 通过 CLI 捕获 stdout
	old := os.Stdout
	r, w, _ := os.Pipe()
	os.Stdout = w
	err := cmdLessons([]string{"search", "--query", query})
	w.Close()
	os.Stdout = old
	if err != nil {
		t.Fatal(err)
	}
	var buf bytes.Buffer
	if _, err := buf.ReadFrom(r); err != nil {
		t.Fatal(err)
	}
	var hits []lessonHit
	for _, line := range strings.Split(strings.TrimSpace(buf.String()), "\n") {
		if line == "" {
			continue
		}
		var h lessonHit
		if err := json.Unmarshal([]byte(line), &h); err != nil {
			t.Fatalf("bad json %q: %v", line, err)
		}
		hits = append(hits, h)
	}
	return hits
}

func listTestLessons(t *testing.T, task string) []lessonRecord {
	t.Helper()
	old := os.Stdout
	r, w, _ := os.Pipe()
	os.Stdout = w
	err := cmdLessons([]string{"list", "--task", task})
	w.Close()
	os.Stdout = old
	if err != nil {
		t.Fatal(err)
	}
	var buf bytes.Buffer
	if _, err := buf.ReadFrom(r); err != nil {
		t.Fatal(err)
	}
	var recs []lessonRecord
	for _, line := range strings.Split(strings.TrimSpace(buf.String()), "\n") {
		if line == "" {
			continue
		}
		var rec lessonRecord
		if err := json.Unmarshal([]byte(line), &rec); err != nil {
			t.Fatalf("bad json %q: %v", line, err)
		}
		recs = append(recs, rec)
	}
	return recs
}

func TestLessonsAddRequiresFlags(t *testing.T) {
	lessonsTestPath(t)
	if err := cmdLessons([]string{"add", "--task", "x"}); err == nil {
		t.Fatal("want usage error for missing --lesson")
	}
	if err := cmdLessons([]string{"frobnicate"}); err == nil {
		t.Fatal("want usage error for unknown subcommand")
	}
}
