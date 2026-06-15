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

func TestContribMarginSpaceBinary(t *testing.T) {
	path := filepath.Join("..", "testdata", "xgboost_smoke.json")
	m, err := io.LoadFromFile(path, &io.LoadOptions{LoadTransformation: true})
	if err != nil {
		t.Fatal(err)
	}
	nFeat := m.NFeatures()
	cols := nFeat + 1
	x := make([]float64, nFeat)
	x[1] = 0.7
	x[5] = 0.6

	contrib := make([]float64, cols)
	if err := m.PredictWithRequest(predict.Request{
		Matrix: predict.DenseMatrix{Values: x, Rows: 1, Cols: nFeat},
		Output: predict.OutputContribution,
	}, contrib); err != nil {
		t.Fatal(err)
	}
	margin := make([]float64, 1)
	if err := m.PredictWithRequest(predict.Request{
		Matrix: predict.DenseMatrix{Values: x, Rows: 1, Cols: nFeat},
		Output: predict.OutputMargin,
	}, margin); err != nil {
		t.Fatal(err)
	}
	value := make([]float64, 1)
	if err := m.PredictWithRequest(predict.Request{
		Matrix: predict.DenseMatrix{Values: x, Rows: 1, Cols: nFeat},
		Output: predict.OutputValue,
	}, value); err != nil {
		t.Fatal(err)
	}

	sum := 0.0
	for _, v := range contrib {
		sum += v
	}
	if math.Abs(sum-margin[0]) > 1e-6 {
		t.Errorf("contrib sum=%f margin=%f", sum, margin[0])
	}
	if math.Abs(margin[0]-value[0]) < 1e-6 {
		t.Fatal("expected margin != value for binary:logistic")
	}
}

func TestContribBiasIsExpectedValueNotBaseScore(t *testing.T) {
	// leaves：末列 bias = 全零背景 margin（ExpectedValue），非 ForestIR.BaseScore。
	// XGBoost Python：末列为上游固定 bias 分解（见 shap_contribs_smoke.tsv）。
	path := filepath.Join("..", "testdata", "xgboost_smoke.json")
	m, err := io.LoadFromFile(path, &io.LoadOptions{LoadTransformation: false})
	if err != nil {
		t.Fatal(err)
	}
	f := m.Forest()
	if f == nil {
		t.Fatal("nil forest")
	}
	exp := m.Explain()
	nFeat := m.NFeatures()
	cols := nFeat + 1
	x := make([]float64, nFeat)
	x[1] = 0.7

	out := make([]float64, cols)
	if err := m.PredictWithRequest(predict.Request{
		Matrix: predict.DenseMatrix{Values: x, Rows: 1, Cols: nFeat},
		Output: predict.OutputContribution,
	}, out); err != nil {
		t.Fatal(err)
	}
	bias := out[nFeat]
	ev := exp.ExpectedValue()
	if math.Abs(bias-ev) > 1e-9 {
		t.Errorf("bias=%f ExpectedValue=%f", bias, ev)
	}
	if math.Abs(f.BaseScore) > 1e-9 {
		t.Logf("BaseScore=%f (logit 后)；bias 仍取 ExpectedValue=%f", f.BaseScore, ev)
	}

	goldenPath := filepath.Join("..", "testdata", "shap_contribs_smoke.tsv")
	_, _, xgbGold := loadShapContribsGolden(t, goldenPath)
	xgbBias := xgbGold[0][nFeat]
	if math.Abs(bias-xgbBias) < 1e-3 {
		t.Fatal("expected leaves bias to differ from XGBoost pred_contribs bias column")
	}
}

func TestContribMulticlassLeavesBitExact(t *testing.T) {
	modelPath := filepath.Join("..", "testdata", "xgboost_multiclass_smoke.json")
	goldenPath := filepath.Join("..", "testdata", "shap_contribs_multiclass_leaves.tsv")

	m, err := io.LoadFromFile(modelPath, &io.LoadOptions{LoadTransformation: false})
	if err != nil {
		t.Fatal(err)
	}
	samples, margins, gold := loadMulticlassContribGolden(t, goldenPath)
	nFeat := m.NFeatures()
	nGroups := m.NRawOutputGroups()
	cols := nFeat + 1

	for si, x := range samples {
		out := make([]float64, nGroups*cols)
		if err := m.PredictWithRequest(predict.Request{
			Matrix: predict.DenseMatrix{Values: x, Rows: 1, Cols: nFeat},
			Output: predict.OutputContribution,
		}, out); err != nil {
			t.Fatalf("sample %d: %v", si, err)
		}
		for k := 0; k < nGroups; k++ {
			off := k * cols
			for fi := 0; fi < cols; fi++ {
				want, ok := gold[si][k][fi]
				if !ok {
					t.Fatalf("sample %d class %d missing feat %d", si, k, fi)
				}
				if math.Abs(out[off+fi]-want) > 1e-6 {
					t.Errorf("sample %d class %d feat %d: got %.10f want %.10f",
						si, k, fi, out[off+fi], want)
				}
			}
			sum := 0.0
			for fi := 0; fi < cols; fi++ {
				sum += out[off+fi]
			}
			if math.Abs(sum-margins[si][k]) > 1e-6 {
				t.Errorf("sample %d class %d additivity: sum=%f margin=%f", si, k, sum, margins[si][k])
			}
		}
	}
}

func TestContribMulticlassXGBoostSemantic(t *testing.T) {
	modelPath := filepath.Join("..", "testdata", "xgboost_multiclass_smoke.json")
	goldenPath := filepath.Join("..", "testdata", "shap_contribs_multiclass_smoke.tsv")

	m, err := io.LoadFromFile(modelPath, &io.LoadOptions{LoadTransformation: false})
	if err != nil {
		t.Fatal(err)
	}
	samples, margins, xgbGold := loadMulticlassContribGolden(t, goldenPath)
	nFeat := m.NFeatures()
	nGroups := m.NRawOutputGroups()
	cols := nFeat + 1

	for si, x := range samples {
		margin := make([]float64, nGroups)
		if err := m.PredictWithRequest(predict.Request{
			Matrix: predict.DenseMatrix{Values: x, Rows: 1, Cols: nFeat},
			Output: predict.OutputMargin,
		}, margin); err != nil {
			t.Fatal(err)
		}
		for k := 0; k < nGroups; k++ {
			if math.Abs(margin[k]-margins[si][k]) > 1e-4 {
				t.Errorf("sample %d class %d margin: got %f want %f", si, k, margin[k], margins[si][k])
			}
			xgbSum := 0.0
			for fi := 0; fi < cols; fi++ {
				xgbSum += xgbGold[si][k][fi]
			}
			if math.Abs(xgbSum-margins[si][k]) > 1e-4 {
				t.Errorf("sample %d class %d xgb additivity: sum=%f margin=%f", si, k, xgbSum, margins[si][k])
			}
		}
	}
}

func TestContribMulticlassLGBMAdditivity(t *testing.T) {
	path := filepath.Join("..", "testdata", "lgmulticlass.model")
	m, err := io.LoadFromFile(path, &io.LoadOptions{LoadTransformation: false})
	if err != nil {
		t.Fatal(err)
	}
	nFeat := m.NFeatures()
	nGroups := m.NRawOutputGroups()
	cols := nFeat + 1

	x := loadMulticlassTestRow(t, filepath.Join("..", "testdata", "multiclass_test.tsv"), 0)

	out := make([]float64, nGroups*cols)
	if err := m.PredictWithRequest(predict.Request{
		Matrix: predict.DenseMatrix{Values: x, Rows: 1, Cols: nFeat},
		Output: predict.OutputContribution,
	}, out); err != nil {
		t.Fatal(err)
	}
	margin := make([]float64, nGroups)
	if err := m.PredictWithRequest(predict.Request{
		Matrix: predict.DenseMatrix{Values: x, Rows: 1, Cols: nFeat},
		Output: predict.OutputMargin,
	}, margin); err != nil {
		t.Fatal(err)
	}
	bases := m.Explain().ExpectedValues()
	for k := 0; k < nGroups; k++ {
		off := k * cols
		sum := 0.0
		for fi := 0; fi < cols; fi++ {
			sum += out[off+fi]
		}
		if math.Abs(sum-margin[k]) > 1e-2 {
			t.Errorf("class %d additivity: sum=%f margin=%f", k, sum, margin[k])
		}
		if math.Abs(out[off+cols-1]-bases[k]) > 1e-6 {
			t.Errorf("class %d bias=%f ExpectedValues=%f", k, out[off+cols-1], bases[k])
		}
	}
}

func multiclassSmokeSamples() [][]float64 {
	return [][]float64{
		{1.20330803, 2.52165753, 1.23475345, 1.28560822, 0.71941107, -1.91720997},
		{-0.16262726, -1.78135965, 0.33902813, -0.25846291, -0.70987622, -1.07404013},
	}
}

func loadMulticlassContribGolden(t *testing.T, path string) (
	samples [][]float64,
	margins [][]float64,
	gold []map[int]map[int]float64,
) {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	nSamples := 2
	nGroups := 3
	samples = multiclassSmokeSamples()
	margins = make([][]float64, nSamples)
	gold = make([]map[int]map[int]float64, nSamples)
	for i := range margins {
		margins[i] = make([]float64, nGroups)
		gold[i] = make(map[int]map[int]float64)
	}

	sc := bufio.NewScanner(f)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			if strings.HasPrefix(line, "# margin") {
				parts := strings.Fields(line)
				if len(parts) >= 5 {
					si, _ := strconv.Atoi(parts[2])
					k, _ := strconv.Atoi(parts[3])
					mv, _ := strconv.ParseFloat(parts[4], 64)
					if si >= 0 && si < len(margins) && k >= 0 && k < nGroups {
						margins[si][k] = mv
					}
				}
			}
			continue
		}
		parts := strings.Split(line, "\t")
		if len(parts) != 4 {
			continue
		}
		si, _ := strconv.Atoi(parts[0])
		k, _ := strconv.Atoi(parts[1])
		fi, _ := strconv.Atoi(parts[2])
		v, _ := strconv.ParseFloat(parts[3], 64)
		if si < 0 || si >= len(gold) {
			continue
		}
		if gold[si][k] == nil {
			gold[si][k] = make(map[int]float64)
		}
		gold[si][k][fi] = v
	}
	if err := sc.Err(); err != nil {
		t.Fatal(err)
	}
	return samples, margins, gold
}

func loadMulticlassTestRow(t *testing.T, path string, row int) []float64 {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	lines := strings.Split(strings.TrimSpace(string(data)), "\n")
	if row >= len(lines) {
		t.Fatalf("row %d out of range", row)
	}
	parts := strings.Split(lines[row], "\t")
	if len(parts) < 2 {
		t.Fatal("bad tsv row")
	}
	x := make([]float64, len(parts)-1)
	for i := 1; i < len(parts); i++ {
		x[i-1], err = strconv.ParseFloat(parts[i], 64)
		if err != nil {
			t.Fatal(err)
		}
	}
	return x
}
