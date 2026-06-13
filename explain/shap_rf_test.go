package explain_test

import (
	"bufio"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/dmitryikh/leaves/io"
	"github.com/dmitryikh/leaves/tree"
)

func TestXGBoostRFGolden(t *testing.T) {
	modelPath := filepath.Join("..", "testdata", "xgboost_rf_smoke.json")
	predPath := filepath.Join("..", "testdata", "xgboost_rf_smoke_pred.txt")

	m, err := io.LoadFromFile(modelPath, &io.LoadOptions{LoadTransformation: false})
	if err != nil {
		t.Fatal(err)
	}
	forest := m.Forest()
	if forest == nil {
		t.Fatal("nil forest")
	}
	if forest.NumParallelTree < 2 {
		t.Fatalf("expected NumParallelTree>=2, got %d", forest.NumParallelTree)
	}
	if len(forest.IterationIndptr) < 2 {
		t.Fatal("expected IterationIndptr")
	}

	samples, wantMargin := loadRFPredGolden(t, predPath)

	for si, x := range samples {
		fm := tree.ForestMargin(forest, x, 0)
		if math.Abs(fm-wantMargin[si]) > 1e-4 {
			t.Errorf("sample %d ForestMargin: got %f want %f", si, fm, wantMargin[si])
		}
		got := m.PredictSingle(x, 0)
		if math.Abs(got-wantMargin[si]) > 1e-4 {
			t.Errorf("sample %d PredictSingle: got %f want %f", si, got, wantMargin[si])
		}
	}
}

func loadRFPredGolden(t *testing.T, path string) (samples [][]float64, margins []float64) {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	sc := bufio.NewScanner(f)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		parts := strings.Split(line, "\t")
		if len(parts) < 7 {
			continue
		}
		margin, _ := strconv.ParseFloat(parts[1], 64)
		x := make([]float64, 4)
		for i := 0; i < 4; i++ {
			x[i], _ = strconv.ParseFloat(parts[3+i], 64)
		}
		samples = append(samples, x)
		margins = append(margins, margin)
	}
	if err := sc.Err(); err != nil {
		t.Fatal(err)
	}
	if len(samples) == 0 {
		t.Fatal("no golden samples")
	}
	return samples, margins
}
