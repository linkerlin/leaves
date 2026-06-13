package explain_test

import (
	"math"
	"path/filepath"
	"strings"
	"testing"

	_ "github.com/dmitryikh/leaves"
	"github.com/dmitryikh/leaves/explain"
	"github.com/dmitryikh/leaves/io"
)

func TestDumpTextSmoke(t *testing.T) {
	path := filepath.Join("..", "testdata", "xgboost_smoke.json")
	m, err := io.LoadFromFile(path, &io.LoadOptions{LoadTransformation: false})
	if err != nil {
		t.Fatal(err)
	}
	f := m.Forest()
	if f == nil {
		t.Fatal("expected forest")
	}
	text := explain.DumpText(f, nil)
	if !strings.Contains(text, "booster[0]:") {
		t.Fatalf("unexpected dump: %s", text)
	}
	if !strings.Contains(text, "0:[f0<=") || !strings.Contains(text, "yes=") {
		t.Fatalf("expected XGBoost-style dump: %s", text)
	}
}

func TestDumpJSONSmoke(t *testing.T) {
	path := filepath.Join("..", "testdata", "xgboost_smoke.json")
	m, err := io.LoadFromFile(path, &io.LoadOptions{LoadTransformation: false})
	if err != nil {
		t.Fatal(err)
	}
	raw, err := explain.DumpJSON(m.Forest(), nil)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(raw), `"num_trees"`) || !strings.Contains(string(raw), `"split_feature"`) {
		t.Fatalf("unexpected json: %s", raw)
	}
}

func TestDumpDOTSmoke(t *testing.T) {
	path := filepath.Join("..", "testdata", "xgboost_smoke.json")
	m, err := io.LoadFromFile(path, &io.LoadOptions{LoadTransformation: false})
	if err != nil {
		t.Fatal(err)
	}
	dot := explain.DumpDOT(m.Forest(), nil)
	if !strings.Contains(dot, "digraph forest") || !strings.Contains(dot, "subgraph cluster_0") {
		t.Fatalf("unexpected dot: %s", dot)
	}
}

func TestImportanceWeight(t *testing.T) {
	path := filepath.Join("..", "testdata", "lg_breast_cancer.txt")
	m, err := io.LoadFromFile(path, io.DefaultLoadOptions())
	if err != nil {
		t.Fatal(err)
	}
	f := m.Forest()
	if f == nil {
		t.Fatal("expected forest")
	}
	imp := explain.ComputeImportance(f, explain.ImportanceWeight, nil)
	if imp == nil || len(imp.Scores) == 0 {
		t.Fatal("empty importance")
	}
	total := 0.0
	for _, s := range imp.Scores {
		total += s
	}
	if total <= 0 {
		t.Fatal("expected positive total weight")
	}
}

func TestImportanceTotalGain(t *testing.T) {
	path := filepath.Join("..", "testdata", "xgboost_smoke.json")
	m, err := io.LoadFromFile(path, &io.LoadOptions{LoadTransformation: false})
	if err != nil {
		t.Fatal(err)
	}
	imp := explain.ComputeImportance(m.Forest(), explain.ImportanceTotalGain, nil)
	if imp == nil {
		t.Fatal("nil importance")
	}
	sum := 0.0
	for _, s := range imp.Scores {
		sum += s
	}
	if math.Abs(sum-1.0) > 1e-6 {
		t.Errorf("total gain should sum to 1, got %f", sum)
	}
}
