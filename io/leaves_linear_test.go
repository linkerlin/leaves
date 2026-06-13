package io_test

import (
	"bytes"
	"math"
	"testing"

	"github.com/dmitryikh/leaves/io"
	"github.com/dmitryikh/leaves/linear"
	"github.com/dmitryikh/leaves/model"
)

func TestLeavesJSONLinearRoundTrip(t *testing.T) {
	ir := &model.ModelIR{
		Kind:             model.KindGBLinear,
		NumFeatures:      2,
		NRawOutputGroups: 1,
		NOutputGroups:    1,
		Name:             "leaves.gblinear",
		Linear: &linear.LinearIR{
			NumFeatures:     2,
			NumOutputGroups: 1,
			BaseScore:       0.5,
			Weights:         []float64{1.0, 2.0, 0.1},
			Name:            "leaves.gblinear",
		},
	}
	var buf bytes.Buffer
	if err := io.SaveLeavesJSON(&buf, ir, "reg:squarederror"); err != nil {
		t.Fatal(err)
	}
	got, obj, err := io.ParseLeavesJSON(bytes.NewReader(buf.Bytes()))
	if err != nil {
		t.Fatal(err)
	}
	if obj != "reg:squarederror" || got.Linear == nil {
		t.Fatal("parse failed")
	}
	x := []float64{1.0, 2.0}
	// base + w0*x0 + w1*x1 + bias
	want := 0.5 + 1.0 + 4.0 + 0.1
	sum := got.Linear.BaseScore + got.Linear.Weights[2]
	sum += x[0]*got.Linear.Weights[0] + x[1]*got.Linear.Weights[1]
	if math.Abs(sum-want) > 1e-9 {
		t.Fatalf("pred %f want %f", sum, want)
	}
}
