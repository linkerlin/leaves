//go:build ignore

// 生成 testdata/shap_contribs_leaves.tsv（leaves predict.Request 回归黄金值）。
// 用法：go run testdata/gen_shap_leaves_golden.go
package main

import (
	"fmt"
	"os"
	"path/filepath"

	_ "github.com/linkerlin/leaves"
	"github.com/linkerlin/leaves/io"
	"github.com/linkerlin/leaves/predict"
)

func main() {
	root := "."
	if _, err := os.Stat(filepath.Join(root, "go.mod")); err != nil {
		root = filepath.Join("..")
	}
	modelPath := filepath.Join(root, "testdata", "xgboost_smoke.json")
	outPath := filepath.Join(root, "testdata", "shap_contribs_leaves.tsv")

	m, err := io.LoadFromFile(modelPath, &io.LoadOptions{LoadTransformation: false})
	if err != nil {
		panic(err)
	}
	nFeat := m.NFeatures()
	cols := nFeat + 1
	samples := [][]float64{
		func() []float64 { x := make([]float64, nFeat); x[1] = 0.7; x[5] = 0.6; return x }(),
		func() []float64 { x := make([]float64, nFeat); x[0] = 0.3; x[2] = 0.8; return x }(),
		func() []float64 { x := make([]float64, nFeat); x[3] = 0.5; return x }(),
	}

	f, err := os.Create(outPath)
	if err != nil {
		panic(err)
	}
	defer f.Close()

	fmt.Fprintf(f, "# leaves OutputContribution golden (self-regression)\n")
	fmt.Fprintf(f, "# n_samples=%d n_cols=%d n_features=%d\n", len(samples), cols, nFeat)

	for si, x := range samples {
		margin := make([]float64, 1)
		if err := m.PredictWithRequest(predict.Request{
			Matrix: predict.DenseMatrix{Values: x, Rows: 1, Cols: nFeat},
			Output: predict.OutputMargin,
		}, margin); err != nil {
			panic(err)
		}
		out := make([]float64, cols)
		if err := m.PredictWithRequest(predict.Request{
			Matrix: predict.DenseMatrix{Values: x, Rows: 1, Cols: nFeat},
			Output: predict.OutputContribution,
		}, out); err != nil {
			panic(err)
		}
		fmt.Fprintf(f, "# margin %d %.10f\n", si, margin[0])
		for fi := 0; fi < cols; fi++ {
			fmt.Fprintf(f, "%d\t%d\t%.10f\n", si, fi, out[fi])
		}
	}
	fmt.Println("wrote", outPath)
}
