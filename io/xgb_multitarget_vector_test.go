package io_test

import (
	"bufio"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/linkerlin/leaves/io"
	"github.com/linkerlin/leaves/tree"
)

func TestXGBoostMultiTargetVectorE2E(t *testing.T) {
	modelPath := filepath.Join("..", "testdata", "xgboost_multitarget_vector.json")
	predPath := filepath.Join("..", "testdata", "xgboost_multitarget_vector_pred.txt")

	m, err := io.LoadFromFile(modelPath, &io.LoadOptions{LoadTransformation: false})
	if err != nil {
		t.Fatal(err)
	}
	forest := m.Forest()
	if forest == nil {
		t.Fatal("nil forest")
	}
	if forest.NumOutputGroups != 2 {
		t.Fatalf("NumOutputGroups=%d want 2", forest.NumOutputGroups)
	}
	if len(forest.Trees) == 0 || forest.Trees[0].OutputDim != 2 {
		t.Fatalf("expected vector leaf trees OutputDim=2, got %d", forest.Trees[0].OutputDim)
	}

	samples, want := loadMultiTargetGolden(t, predPath)

	for si, x := range samples {
		margins := tree.ForestMargins(forest, x, 0)
		if len(margins) != 2 {
			t.Fatalf("sample %d: margins len %d", si, len(margins))
		}
		for k := 0; k < 2; k++ {
			if math.Abs(margins[k]-want[si][k]) > 1e-4 {
				t.Errorf("sample %d target %d: ForestMargins got %f want %f", si, k, margins[k], want[si][k])
			}
		}

		raw := make([]float64, m.NRawOutputGroups())
		if err := m.Engine().Predict(x, 0, raw); err != nil {
			t.Fatalf("sample %d engine predict: %v", si, err)
		}
		for k := 0; k < 2; k++ {
			if math.Abs(raw[k]-want[si][k]) > 1e-4 {
				t.Errorf("sample %d target %d: engine got %f want %f", si, k, raw[k], want[si][k])
			}
		}
	}

	bornEng, err := tree.NewBornEngine(forest, tree.ApplyTransformRaw, tree.TransformRaw, 2, nil)
	if err != nil {
		t.Skipf("born: %v", err)
	}
	defer bornEng.Close()
	for si, x := range samples {
		out := make([]float64, 2)
		if err := bornEng.Predict(x, 0, out); err != nil {
			t.Fatalf("born sample %d: %v", si, err)
		}
		for k := 0; k < 2; k++ {
			if math.Abs(out[k]-want[si][k]) > 1e-4 {
				t.Errorf("born sample %d target %d: got %f want %f", si, k, out[k], want[si][k])
			}
		}
	}
}

func loadMultiTargetGolden(t *testing.T, path string) (samples [][]float64, margins [][]float64) {
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
		if len(parts) < 6 {
			continue
		}
		m0, _ := strconv.ParseFloat(parts[1], 64)
		m1, _ := strconv.ParseFloat(parts[2], 64)
		x := make([]float64, 3)
		for i := 0; i < 3; i++ {
			x[i], _ = strconv.ParseFloat(parts[3+i], 64)
		}
		samples = append(samples, x)
		margins = append(margins, []float64{m0, m1})
	}
	if err := sc.Err(); err != nil {
		t.Fatal(err)
	}
	if len(samples) == 0 {
		t.Fatal("no golden samples")
	}
	return samples, margins
}
