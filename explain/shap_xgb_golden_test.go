package explain_test

import (
	"bufio"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/linkerlin/leaves/io"
)

func TestXGBoostPredContribsSemantic(t *testing.T) {
	// XGBoost pred_contribs：末列 bias 为上游分解；leaves 末列为背景 margin。
	// 双方均可加性还原 margin，但逐元素 SHAP 不对齐。
	modelPath := filepath.Join("..", "testdata", "xgboost_smoke.json")
	goldenPath := filepath.Join("..", "testdata", "shap_contribs_smoke.tsv")

	m, err := io.LoadFromFile(modelPath, &io.LoadOptions{LoadTransformation: false})
	if err != nil {
		t.Fatal(err)
	}

	samples, margins, gold := loadShapContribsGolden(t, goldenPath)
	eng := m.Engine()

	for si, x := range samples {
		raw := make([]float64, m.NRawOutputGroups())
		if err := eng.Predict(x, 0, raw); err != nil {
			t.Fatalf("sample %d predict: %v", si, err)
		}
		if math.Abs(raw[0]-margins[si]) > 1e-4 {
			t.Errorf("sample %d margin: got %f want %f", si, raw[0], margins[si])
		}

		// XGBoost 黄金文件内部可加性
		xgbSum := 0.0
		for fi := 0; fi < len(x)+1; fi++ {
			xgbSum += gold[si][fi]
		}
		if math.Abs(xgbSum-margins[si]) > 1e-4 {
			t.Errorf("sample %d xgb golden additivity: sum=%f margin=%f", si, xgbSum, margins[si])
		}

		// leaves interventional SHAP 可加性
		exp := m.Explain()
		phi, err := exp.TreeSHAP([][]float64{x})
		if err != nil {
			t.Fatal(err)
		}
		leafSum := exp.ExpectedValue()
		for _, v := range phi[0] {
			leafSum += v
		}
		if math.Abs(leafSum-raw[0]) > 1e-4 {
			t.Errorf("sample %d leaves additivity: sum=%f margin=%f", si, leafSum, raw[0])
		}
	}
}

func loadShapContribsGolden(t *testing.T, path string) (samples [][]float64, margins []float64, gold []map[int]float64) {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	nFeatures := 8
	samples = [][]float64{
		func() []float64 { x := make([]float64, nFeatures); x[1] = 0.7; x[5] = 0.6; return x }(),
		func() []float64 { x := make([]float64, nFeatures); x[0] = 0.3; x[2] = 0.8; return x }(),
		func() []float64 { x := make([]float64, nFeatures); x[3] = 0.5; return x }(),
	}
	margins = make([]float64, len(samples))
	gold = make([]map[int]float64, len(samples))
	for i := range gold {
		gold[i] = make(map[int]float64)
	}

	sc := bufio.NewScanner(f)
	curSample := -1
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			if strings.HasPrefix(line, "# margin") {
				parts := strings.Fields(line)
				if len(parts) >= 4 {
					si, _ := strconv.Atoi(parts[2])
					mv, _ := strconv.ParseFloat(parts[3], 64)
					if si >= 0 && si < len(margins) {
						margins[si] = mv
						curSample = si
					}
				}
			}
			continue
		}
		parts := strings.Split(line, "\t")
		if len(parts) != 3 {
			continue
		}
		si, _ := strconv.Atoi(parts[0])
		fi, _ := strconv.Atoi(parts[1])
		v, _ := strconv.ParseFloat(parts[2], 64)
		if si >= 0 && si < len(gold) {
			gold[si][fi] = v
			curSample = si
		}
	}
	_ = curSample
	if err := sc.Err(); err != nil {
		t.Fatal(err)
	}
	return samples, margins, gold
}
