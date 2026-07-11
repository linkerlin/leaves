package transformation

import (
	"math"
	"testing"
)

func approx(want, got []float64, eps float64) bool {
	if len(want) != len(got) {
		return false
	}
	for i := range want {
		if math.Abs(want[i]-got[i]) > eps {
			return false
		}
	}
	return true
}

func TestTransformRaw(t *testing.T) {
	tr := &TransformRaw{NumOutputGroups: 3}
	out := make([]float64, 5)
	if err := tr.Transform([]float64{1.5, -2, 3}, out, 1); err != nil {
		t.Fatal(err)
	}
	want := []float64{0, 1.5, -2, 3, 0}
	if !approx(want, out, 1e-12) {
		t.Fatalf("got %v want %v", out, want)
	}
	if tr.Type() != Raw || tr.Name() != "raw" || tr.NOutputGroups() != 3 {
		t.Fatalf("meta mismatch: type=%v name=%q groups=%d", tr.Type(), tr.Name(), tr.NOutputGroups())
	}
}

func TestTransformLogistic(t *testing.T) {
	tr := &TransformLogistic{}
	out := make([]float64, 2)
	if err := tr.Transform([]float64{0}, out, 1); err != nil {
		t.Fatal(err)
	}
	if math.Abs(out[1]-0.5) > 1e-12 {
		t.Fatalf("sigmoid(0)=%v want 0.5", out[1])
	}
	if err := tr.Transform([]float64{0, 1}, out, 0); err == nil {
		t.Fatal("expected error for len(rawPredictions)!=1")
	}
	if tr.Type() != Logistic || tr.Name() != "logistic" || tr.NOutputGroups() != 1 {
		t.Fatalf("meta mismatch")
	}
}

func TestTransformSoftmax(t *testing.T) {
	tr := &TransformSoftmax{NClasses: 3}
	out := make([]float64, 3)
	if err := tr.Transform([]float64{1, 2, 3}, out, 0); err != nil {
		t.Fatal(err)
	}
	want := []float64{0.09003057317038046, 0.24472847105479764, 0.6652409557748219}
	if !approx(want, out, 1e-9) {
		t.Fatalf("softmax got %v want %v", out, want)
	}
	sum := 0.0
	for _, v := range out {
		sum += v
	}
	if math.Abs(sum-1) > 1e-12 {
		t.Fatalf("softmax sum=%v want 1", sum)
	}
	if err := tr.Transform([]float64{1, 2}, out, 0); err == nil {
		t.Fatal("expected error for len(rawPredictions)!=NClasses")
	}
	if tr.Type() != Softmax || tr.Name() != "softmax" || tr.NOutputGroups() != 3 {
		t.Fatalf("meta mismatch")
	}
}

func TestTransformExponential(t *testing.T) {
	tr := &TransformExponential{}
	out := make([]float64, 1)
	if err := tr.Transform([]float64{1}, out, 0); err != nil {
		t.Fatal(err)
	}
	if math.Abs(out[0]-math.E) > 1e-12 {
		t.Fatalf("exp(1)=%v want %v", out[0], math.E)
	}
	if err := tr.Transform([]float64{1, 2}, out, 0); err == nil {
		t.Fatal("expected error for len(rawPredictions)!=1")
	}
	// Regression: Name() previously returned "logistic" (masked an array OOR panic).
	if tr.Type() != Exponential || tr.Name() != "exponential" || tr.NOutputGroups() != 1 {
		t.Fatalf("meta mismatch: type=%v name=%q groups=%d", tr.Type(), tr.Name(), tr.NOutputGroups())
	}
}

func TestTransformLeafIndex(t *testing.T) {
	tr := &TransformLeafIndex{NumOutputGroups: 2}
	out := make([]float64, 4)
	if err := tr.Transform([]float64{3, 7}, out, 1); err != nil {
		t.Fatal(err)
	}
	want := []float64{0, 3, 7, 0}
	if !approx(want, out, 0) {
		t.Fatalf("got %v want %v", out, want)
	}
	if tr.Type() != LeafIndex || tr.Name() != "leaf_index" || tr.NOutputGroups() != 2 {
		t.Fatalf("meta mismatch")
	}
}

// TestTransformTypeName locks the regression where TransformType(Exponential).Name()
// panicked: transformNames had only 4 entries but Exponential=Last=4 (index OOR).
func TestTransformTypeName(t *testing.T) {
	cases := []struct {
		t    TransformType
		name string
	}{
		{Raw, "raw"},
		{Logistic, "logistic"},
		{Softmax, "softmax"},
		{LeafIndex, "leaf_index"},
		{Exponential, "exponential"},
		{Last, "exponential"},
		{Last + 1, "unknown"},
		{-1, "unknown"},
	}
	for _, c := range cases {
		if got := c.t.Name(); got != c.name {
			t.Errorf("TransformType(%d).Name() = %q want %q", c.t, got, c.name)
		}
	}
}
