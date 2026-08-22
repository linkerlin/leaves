package io_test

import (
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/linkerlin/leaves/v2/io"
)

func writeTemp(t *testing.T, data []byte, name string) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), name)
	if err := os.WriteFile(p, data, 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

func sigmoid(x float64) float64 { return 1 / (1 + math.Exp(-x)) }

// 三类 SOFTMAX：x0<=0.5 → raw=[2,0,1]；x0>0.5 → raw=[0,2,1]。
// 期望概率 = softmax(raw)。
func TestONNXClassifier3ClassSoftmax(t *testing.T) {
	path := writeTemp(t, io.SampleONNXClassifier3Class(), "cls3.onnx")
	ens, err := io.LoadONNX(path, &io.LoadOptions{Backend: io.BackendNative})
	if err != nil {
		t.Fatalf("LoadONNX: %v", err)
	}
	if got := ens.NOutputGroups(); got != 3 {
		t.Fatalf("NOutputGroups=%d want 3", got)
	}

	cases := []struct {
		x    float64
		raws []float64
	}{
		{0.4, []float64{2, 0, 1}},
		{0.6, []float64{0, 2, 1}},
	}
	for _, c := range cases {
		preds := make([]float64, 3)
		if err := ens.PredictDense([]float64{c.x}, 1, 1, preds, 0, 1); err != nil {
			t.Fatal(err)
		}
		// softmax 期望
		max := c.raws[0]
		for _, v := range c.raws {
			if v > max {
				max = v
			}
		}
		var sum float64
		exp := make([]float64, 3)
		for i, v := range c.raws {
			exp[i] = math.Exp(v - max)
			sum += exp[i]
		}
		for i := range exp {
			exp[i] /= sum
			if math.Abs(preds[i]-exp[i]) > 1e-5 {
				t.Errorf("x=%.1f group %d: got %g want %g", c.x, i, preds[i], exp[i])
			}
		}
		if s := preds[0] + preds[1] + preds[2]; math.Abs(s-1) > 1e-5 {
			t.Errorf("x=%.1f probs sum=%g want 1", c.x, s)
		}
	}
}

// 二类 LOGISTIC：单输出组；p=sigmoid(raw)。x0<=0.5 → raw=1 → p=sigmoid(1)。
func TestONNXClassifierBinaryLogistic(t *testing.T) {
	path := writeTemp(t, io.SampleONNXClassifierBinary(), "bin.onnx")
	ens, err := io.LoadONNX(path, &io.LoadOptions{Backend: io.BackendNative})
	if err != nil {
		t.Fatalf("LoadONNX: %v", err)
	}
	if got := ens.NOutputGroups(); got != 1 {
		t.Fatalf("NOutputGroups=%d want 1（LOGISTIC 二类坍缩）", got)
	}
	cases := []struct {
		x   float64
		raw float64
	}{
		{0.4, 1.0},
		{0.6, -1.0},
	}
	for _, c := range cases {
		preds := make([]float64, 1)
		if err := ens.PredictDense([]float64{c.x}, 1, 1, preds, 0, 1); err != nil {
			t.Fatal(err)
		}
		want := sigmoid(c.raw)
		if math.Abs(preds[0]-want) > 1e-5 {
			t.Errorf("x=%.1f: got %g want %g", c.x, preds[0], want)
		}
	}
}

// AVERAGE：类内树数折算。两类各 2 棵同参树、AVERAGE → 每叶权重减半。
func TestONNXClassifierAverage(t *testing.T) {
	// class0: 2 棵树（tree 0,1）叶权 [4,0]；class1: 2 棵树（tree 2,3）叶权 [0,4]
	// AVERAGE 语义 = 类内树求平均：x<=0.5 → [4,0]；x>0.5 → [0,4]；NONE 输出 raw。
	ids := []int64{0, 1, 2}
	nodeTreeids := []int64{}
	nodeids := []int64{}
	trues := []int64{}
	falses := []int64{}
	values := []float64{}
	modes := []string{}
	for _, tr := range []int64{0, 1, 2, 3} {
		for _, nid := range ids {
			nodeTreeids = append(nodeTreeids, tr)
			nodeids = append(nodeids, nid)
			trues = append(trues, map[int64]int64{0: 1, 1: 0, 2: 0}[nid])
			falses = append(falses, map[int64]int64{0: 2, 1: 0, 2: 0}[nid])
			values = append(values, map[int64]float64{0: 0.5, 1: 0, 2: 0}[nid])
			modes = append(modes, map[int64]string{0: "BRANCH_LEQ", 1: "LEAF", 2: "LEAF"}[nid])
		}
	}
	raw := io.WriteONNXTreeEnsembleClassifier(
		nodeTreeids, nodeids, make([]int64, 12), trues, falses, values, modes,
		[]int64{0, 0, 1, 1, 2, 2, 3, 3},
		[]int64{1, 2, 1, 2, 1, 2, 1, 2},
		[]int64{0, 0, 0, 0, 1, 1, 1, 1},
		[]float64{4, 0, 4, 0, 0, 4, 0, 4},
		[]int64{0, 1}, "AVERAGE", "NONE",
	)
	path := writeTemp(t, raw, "avg.onnx")
	ens, err := io.LoadONNX(path, &io.LoadOptions{Backend: io.BackendNative})
	if err != nil {
		t.Fatalf("LoadONNX: %v", err)
	}
	if got := ens.NOutputGroups(); got != 2 {
		t.Fatalf("NOutputGroups=%d want 2", got)
	}
	preds := make([]float64, 2)
	if err := ens.PredictDense([]float64{0.4}, 1, 1, preds, 0, 1); err != nil {
		t.Fatal(err)
	}
	if math.Abs(preds[0]-4.0) > 1e-5 || math.Abs(preds[1]-0.0) > 1e-5 {
		t.Errorf("AVERAGE left: got [%g,%g] want [4,0]", preds[0], preds[1])
	}
	if err := ens.PredictDense([]float64{0.6}, 1, 1, preds, 0, 1); err != nil {
		t.Fatal(err)
	}
	if math.Abs(preds[0]-0.0) > 1e-5 || math.Abs(preds[1]-4.0) > 1e-5 {
		t.Errorf("AVERAGE right: got [%g,%g] want [0,4]", preds[0], preds[1])
	}
}

// 非法 post_transform 拒绝（LOGISTIC + 3 类 → 可行动错误）。
func TestONNXClassifierRejects(t *testing.T) {
	raw := io.WriteONNXTreeEnsembleClassifier(
		[]int64{0, 0, 0}, []int64{0, 1, 2}, []int64{0, 0, 0},
		[]int64{1, 0, 0}, []int64{2, 0, 0},
		[]float64{0.5, 0, 0}, []string{"BRANCH_LEQ", "LEAF", "LEAF"},
		[]int64{0, 0, 1, 1, 2, 2}, []int64{1, 2, 1, 2, 1, 2},
		[]int64{0, 0, 1, 1, 2, 2}, []float64{1, 0, 0, 1, 0, 1},
		[]int64{0, 1, 2}, "SUM", "LOGISTIC", // 3 类配 LOGISTIC：非法
	)
	path := writeTemp(t, raw, "bad.onnx")
	_, err := io.LoadONNX(path, &io.LoadOptions{Backend: io.BackendNative})
	if err == nil {
		t.Fatal("expected LOGISTIC/3-class rejection")
	}
	if !strings.Contains(err.Error(), "LOGISTIC") {
		t.Fatalf("error not actionable: %v", err)
	}
}
