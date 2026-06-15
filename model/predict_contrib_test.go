package model_test

import (
	"bufio"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	_ "github.com/linkerlin/leaves"
	"github.com/linkerlin/leaves/io"
	"github.com/linkerlin/leaves/predict"
)

func TestPredictWithRequestContribution(t *testing.T) {
	modelPath := filepath.Join("..", "testdata", "xgboost_smoke.json")
	goldenPath := filepath.Join("..", "testdata", "shap_contribs_smoke.tsv")

	m, err := io.LoadFromFile(modelPath, &io.LoadOptions{LoadTransformation: false})
	if err != nil {
		t.Fatal(err)
	}

	samples, margins, gold := loadShapContribsGolden(t, goldenPath)
	nFeat := m.NFeatures()
	cols := nFeat + 1

	for si, x := range samples {
		out := make([]float64, cols)
		err := m.PredictWithRequest(predict.Request{
			Matrix: predict.DenseMatrix{Values: x, Rows: 1, Cols: nFeat},
			Output: predict.OutputContribution,
		}, out)
		if err != nil {
			t.Fatalf("sample %d: %v", si, err)
		}

		sum := 0.0
		for fi := 0; fi < cols; fi++ {
			sum += out[fi]
		}
		if math.Abs(sum-margins[si]) > 1e-4 {
			t.Errorf("sample %d additivity: sum=%f margin=%f", si, sum, margins[si])
		}

		xgbSum := 0.0
		for fi := 0; fi < cols; fi++ {
			xgbSum += gold[si][fi]
		}
		if math.Abs(xgbSum-margins[si]) > 1e-4 {
			t.Errorf("sample %d xgb golden additivity: sum=%f margin=%f", si, xgbSum, margins[si])
		}
	}
}

// TestPredictContribsLeavesBitExact 对 leaves 自生成黄金值逐元素回归（非 XGBoost 逐元素对齐）。
func TestPredictContribsLeavesBitExact(t *testing.T) {
	modelPath := filepath.Join("..", "testdata", "xgboost_smoke.json")
	goldenPath := filepath.Join("..", "testdata", "shap_contribs_leaves.tsv")

	m, err := io.LoadFromFile(modelPath, &io.LoadOptions{LoadTransformation: false})
	if err != nil {
		t.Fatal(err)
	}

	samples, margins, gold := loadShapContribsGolden(t, goldenPath)
	nFeat := m.NFeatures()
	cols := nFeat + 1

	for si, x := range samples {
		out := make([]float64, cols)
		if err := m.PredictWithRequest(predict.Request{
			Matrix: predict.DenseMatrix{Values: x, Rows: 1, Cols: nFeat},
			Output: predict.OutputContribution,
		}, out); err != nil {
			t.Fatalf("sample %d: %v", si, err)
		}
		for fi := 0; fi < cols; fi++ {
			want, ok := gold[si][fi]
			if !ok {
				t.Fatalf("sample %d missing golden feature %d", si, fi)
			}
			if math.Abs(out[fi]-want) > 1e-9 {
				t.Errorf("sample %d feat %d: got %.10f want %.10f", si, fi, out[fi], want)
			}
		}
		margin := make([]float64, 1)
		if err := m.PredictWithRequest(predict.Request{
			Matrix: predict.DenseMatrix{Values: x, Rows: 1, Cols: nFeat},
			Output: predict.OutputMargin,
		}, margin); err != nil {
			t.Fatal(err)
		}
		if math.Abs(margin[0]-margins[si]) > 1e-9 {
			t.Errorf("sample %d margin: got %.10f want %.10f", si, margin[0], margins[si])
		}
	}
}

func TestPredictWithRequestApproxContribution(t *testing.T) {
	path := filepath.Join("..", "testdata", "lg_breast_cancer.txt")
	m, err := io.LoadFromFile(path, io.DefaultLoadOptions())
	if err != nil {
		t.Fatal(err)
	}
	nFeat := m.NFeatures()
	x := make([]float64, nFeat)
	out := make([]float64, nFeat+1)
	if err := m.PredictWithRequest(predict.Request{
		Matrix: predict.DenseMatrix{Values: x, Rows: 1, Cols: nFeat},
		Output: predict.OutputApproxContribution,
	}, out); err != nil {
		t.Fatal(err)
	}
	total := 0.0
	for _, v := range out {
		total += v
	}
	if total == 0 {
		t.Fatal("expected non-zero Saabas output")
	}
}

func TestPredictWithRequestInteraction(t *testing.T) {
	path := filepath.Join("..", "testdata", "xgboost_smoke.json")
	m, err := io.LoadFromFile(path, &io.LoadOptions{LoadTransformation: false})
	if err != nil {
		t.Fatal(err)
	}
	nFeat := m.NFeatures()
	cols := nFeat + 1
	x := make([]float64, nFeat)
	x[1] = 0.7
	x[5] = 0.6

	out := make([]float64, cols*cols)
	if err := m.PredictWithRequest(predict.Request{
		Matrix: predict.DenseMatrix{Values: x, Rows: 1, Cols: nFeat},
		Output: predict.OutputInteraction,
	}, out); err != nil {
		t.Fatal(err)
	}

	total := out[cols*cols-1] // bias diagonal
	for fi := 0; fi < cols; fi++ {
		for fj := 0; fj < cols; fj++ {
			if fi == cols-1 && fj == cols-1 {
				continue
			}
			total += out[fi*cols+fj]
		}
	}

	margin := make([]float64, 1)
	if err := m.PredictWithRequest(predict.Request{
		Matrix: predict.DenseMatrix{Values: x, Rows: 1, Cols: nFeat},
		Output: predict.OutputMargin,
	}, margin); err != nil {
		t.Fatal(err)
	}
	if math.Abs(total-margin[0]) > 1e-4 {
		t.Errorf("interaction additivity: total=%f margin=%f", total, margin[0])
	}
}

func TestPredictWithRequestContributionRequiresTree(t *testing.T) {
	path := filepath.Join("..", "testdata", "xgboost_smoke.json")
	m, err := io.LoadFromFile(path, nil)
	if err != nil {
		t.Fatal(err)
	}
	// smoke 为 gbtree，应成功；线性模型另测。此处验证 slice 长度不足报错。
	err = m.PredictWithRequest(predict.Request{
		Matrix: predict.DenseMatrix{Values: make([]float64, m.NFeatures()), Rows: 1, Cols: m.NFeatures()},
		Output: predict.OutputContribution,
	}, make([]float64, 1))
	if err == nil {
		t.Fatal("expected short buffer error")
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
		}
	}
	if err := sc.Err(); err != nil {
		t.Fatal(err)
	}
	return samples, margins, gold
}
