package quantize_test

import (
	"math"
	"testing"

	"github.com/linkerlin/leaves/quantize"
	"github.com/linkerlin/leaves/tree"
)

func TestEncodeDecodeThreshold(t *testing.T) {
	min, span := 0.1, 2.0
	orig := 0.55
	q := quantize.EncodeThresholdForTest(orig, min, span, quantize.Levels)
	got := quantize.DecodeThresholdForTest(q, min, span, quantize.Levels)
	if math.Abs(got-orig) > span/float64(quantize.Levels)+1e-12 {
		t.Fatalf("roundtrip: orig=%v got=%v", orig, got)
	}
}

func TestQuantizeForestDoesNotMutateSource(t *testing.T) {
	tr := tree.BuildTreeIR(nil, []float64{0.2}, nil, nil, 0)
	f := &tree.ForestIR{
		NumFeatures:     1,
		NumOutputGroups: 1,
		Trees:           []tree.TreeIR{*tr},
		BaseScore:       0.5,
	}
	before := f.Trees[0].SplitThreshold
	qf, err := quantize.QuantizeForest(f, quantize.Config{})
	if err != nil {
		t.Fatal(err)
	}
	if len(qf.QThreshold) != 1 {
		t.Fatal("expected quantized thresholds")
	}
	_ = qf // 量化结果独立存储
	if len(f.Trees[0].SplitThreshold) != len(before) {
		t.Fatal("source forest mutated")
	}
}

func TestQuantizedEngineLeafIndicesUnsupported(t *testing.T) {
	tr := tree.BuildTreeIR(nil, []float64{0.2}, nil, nil, 0)
	f := &tree.ForestIR{NumFeatures: 1, NumOutputGroups: 1, Trees: []tree.TreeIR{*tr}}
	qf, err := quantize.QuantizeForest(f, quantize.Config{})
	if err != nil {
		t.Fatal(err)
	}
	eng, err := quantize.NewEngine(qf, nil, tree.TransformRaw, 1)
	if err != nil {
		t.Fatal(err)
	}
	err = eng.PredictLeafIndicesDense([]float64{0}, 1, 1, []float64{0})
	if err == nil {
		t.Fatal("expected leaf indices unsupported")
	}
}
