package contract

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestHashFileStable(t *testing.T) {
	p := filepath.Join(t.TempDir(), "a.tsv")
	if err := os.WriteFile(p, []byte("hello"), 0o644); err != nil {
		t.Fatal(err)
	}
	h1, err := HashFile(p)
	if err != nil {
		t.Fatal(err)
	}
	h2, err := HashFile(p)
	if err != nil {
		t.Fatal(err)
	}
	if h1 != h2 || len(h1) != 64 {
		t.Fatalf("hash not stable: %q vs %q", h1, h2)
	}
	// 已知向量：sha256("hello")
	if h1 != "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824" {
		t.Fatalf("sha256 vector mismatch: %s", h1)
	}
	if _, err := HashFile(filepath.Join(t.TempDir(), "missing")); err == nil {
		t.Fatal("want missing file to fail")
	}
}

func TestFeatureSchemaHashOrderSensitive(t *testing.T) {
	a := FeatureSchemaHash([]string{"f1", "f2"}, nil)
	b := FeatureSchemaHash([]string{"f2", "f1"}, nil)
	if a == b {
		t.Fatal("column order must change hash")
	}
	c := FeatureSchemaHash([]string{"f1", "f2"}, []float64{0, 1})
	if a == c {
		t.Fatal("defaults must change hash")
	}
	d := FeatureSchemaHash([]string{"f1", "f2"}, []float64{0, 2})
	if c == d {
		t.Fatal("default values must change hash")
	}
	if len(a) != 64 {
		t.Fatalf("hash length %d", len(a))
	}
}

type jsonlRow struct {
	ID   string    `json:"id"`
	When time.Time `json:"when"`
}

func TestJSONLRoundTrip(t *testing.T) {
	p := filepath.Join(t.TempDir(), "rows.jsonl")
	rows := []jsonlRow{
		{ID: "a", When: time.Date(2026, 8, 20, 0, 0, 0, 0, time.UTC)},
		{ID: "b", When: time.Date(2026, 8, 21, 0, 0, 0, 0, time.UTC)},
	}
	if err := WriteJSONL(p, rows); err != nil {
		t.Fatal(err)
	}
	got, err := ReadJSONL[jsonlRow](p)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 2 || got[0].ID != "a" || !got[1].When.Equal(rows[1].When) {
		t.Fatalf("round trip mismatch: %+v", got)
	}
	if strings.Count(mustRead(t, p), "\n") != 2 {
		t.Fatal("want one JSON object per line")
	}
}

func TestVerifyFilesMismatch(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "a.tsv")
	if err := os.WriteFile(p, []byte("hello"), 0o644); err != nil {
		t.Fatal(err)
	}
	h, err := HashFile(p)
	if err != nil {
		t.Fatal(err)
	}
	snap := DatasetSnapshot{
		SnapshotID: "s1", SchemaVersion: SchemaVersion,
		CreatedAt:         time.Date(2026, 8, 20, 0, 0, 0, 0, time.UTC),
		Purpose:           "train",
		InputFiles:        []FileRef{{Path: "a.tsv", SHA256: h}},
		FeatureSchemaHash: strings.Repeat("b", 64),
	}
	if err := VerifyFiles(&snap, dir); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(p, []byte("tampered"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := VerifyFiles(&snap, dir); err == nil || !strings.Contains(err.Error(), "mismatch") {
		t.Fatalf("want hash mismatch failure, got %v", err)
	}
}

func mustRead(t *testing.T, p string) string {
	t.Helper()
	b, err := os.ReadFile(p)
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}
