//go:build ignore

// 生成 testdata/shap_contribs_multiclass_leaves.tsv。
// 用法：go run testdata/gen_shap_multiclass_leaves_golden.go
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
	modelPath := filepath.Join(root, "testdata", "xgboost_multiclass_smoke.json")
	outPath := filepath.Join(root, "testdata", "shap_contribs_multiclass_leaves.tsv")

	m, err := io.LoadFromFile(modelPath, &io.LoadOptions{LoadTransformation: false})
	if err != nil {
		panic(err)
	}
	nFeat := m.NFeatures()
	nGroups := m.NRawOutputGroups()
	cols := nFeat + 1

	// 与 gen_shap_multiclass_golden.py 相同：sklearn make_classification 前两行
	samples := multiclassSamples()

	f, err := os.Create(outPath)
	if err != nil {
		panic(err)
	}
	defer f.Close()

	fmt.Fprintf(f, "# leaves multiclass OutputContribution golden\n")
	fmt.Fprintf(f, "# n_samples=%d n_groups=%d n_cols=%d n_features=%d\n",
		len(samples), nGroups, cols, nFeat)

	for si, x := range samples {
		margin := make([]float64, nGroups)
		if err := m.PredictWithRequest(predict.Request{
			Matrix: predict.DenseMatrix{Values: x, Rows: 1, Cols: nFeat},
			Output: predict.OutputMargin,
		}, margin); err != nil {
			panic(err)
		}
		out := make([]float64, nGroups*cols)
		if err := m.PredictWithRequest(predict.Request{
			Matrix: predict.DenseMatrix{Values: x, Rows: 1, Cols: nFeat},
			Output: predict.OutputContribution,
		}, out); err != nil {
			panic(err)
		}
		for k := 0; k < nGroups; k++ {
			fmt.Fprintf(f, "# margin %d %d %.10f\n", si, k, margin[k])
			off := k * cols
			for fi := 0; fi < cols; fi++ {
				fmt.Fprintf(f, "%d\t%d\t%d\t%.10f\n", si, k, fi, out[off+fi])
			}
		}
	}
	fmt.Println("wrote", outPath)
}

func multiclassSamples() [][]float64 {
	return [][]float64{
		{1.20330803, 2.52165753, 1.23475345, 1.28560822, 0.71941107, -1.91720997},
		{-0.16262726, -1.78135965, 0.33902813, -0.25846291, -0.70987622, -1.07404013},
	}
}
