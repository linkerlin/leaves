package quantize_test

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
	"github.com/linkerlin/leaves/quantize"
)

func TestParityXGBoostSmokeGate(t *testing.T) {
	path := filepath.Join("..", "testdata", "xgboost_smoke.json")
	m, err := io.LoadFromFile(path, &io.LoadOptions{LoadTransformation: false})
	if err != nil {
		t.Fatal(err)
	}
	orig := m.Forest()
	if orig == nil {
		t.Fatal("nil forest")
	}
	qf, err := quantize.QuantizeForest(orig, quantize.Config{})
	if err != nil {
		t.Fatal(err)
	}
	rows := loadFeatureRows(t, filepath.Join("..", "testdata", "breast_cancer_test.tsv"), orig.NumFeatures, 32)
	if len(rows) == 0 {
		t.Fatal("no samples")
	}

	gate := quantize.DefaultGate()
	res, err := quantize.CheckParityWithGate(orig, qf, rows, 0, gate)
	if err != nil {
		t.Fatalf("parity gate: %v (max_diff=%.6g mean=%.6g thresh_err=%.6g failures=%d/%d)",
			err, res.MaxMarginDiff, res.MeanMarginDiff, res.MaxThresholdErr, res.Failures, res.Samples)
	}
	t.Logf("parity ok: samples=%d max_diff=%.6g mean=%.6g thresh_err=%.6g",
		res.Samples, res.MaxMarginDiff, res.MeanMarginDiff, res.MaxThresholdErr)
}

func TestQuantizedEngineMatchesMargins(t *testing.T) {
	path := filepath.Join("..", "testdata", "xgboost_smoke.json")
	m, err := io.LoadFromFile(path, &io.LoadOptions{LoadTransformation: false})
	if err != nil {
		t.Fatal(err)
	}
	orig := m.Forest()
	qf, err := quantize.QuantizeForest(orig, quantize.Config{})
	if err != nil {
		t.Fatal(err)
	}
	eng, err := quantize.NewEngine(qf, nil, 0, m.NOutputGroups())
	if err != nil {
		t.Fatal(err)
	}
	x := loadFeatureRows(t, filepath.Join("..", "testdata", "breast_cancer_test.tsv"), orig.NumFeatures, 1)[0]
	qm := quantize.MarginsForTest(qf, x, 0)
	pred := make([]float64, m.NOutputGroups())
	if err := eng.Predict(x, 0, pred); err != nil {
		t.Fatal(err)
	}
	if math.Abs(pred[0]-qm[0]) > 1e-12 {
		t.Fatalf("engine=%v margins=%v", pred[0], qm[0])
	}
}

func loadFeatureRows(t *testing.T, path string, nFeat, maxRows int) [][]float64 {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	var rows [][]float64
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		if maxRows > 0 && len(rows) >= maxRows {
			break
		}
		parts := strings.Fields(sc.Text())
		if len(parts) < nFeat {
			continue
		}
		row := make([]float64, nFeat)
		ok := true
		for i := 0; i < nFeat; i++ {
			row[i], err = strconv.ParseFloat(parts[i], 64)
			if err != nil {
				ok = false
				break
			}
		}
		if ok {
			rows = append(rows, row)
		}
	}
	return rows
}
