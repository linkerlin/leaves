package data

import (
	"math"
	"testing"
)

func TestAFTIntervalValidate(t *testing.T) {
	cases := []struct {
		iv  AFTInterval
		ok  bool
	}{
		{AFTInterval{2, 2}, true},
		{AFTInterval{3, math.Inf(1)}, true},
		{AFTInterval{0, 4}, true},
		{AFTInterval{1, 5}, true},
		{AFTInterval{-1, 2}, false},
		{AFTInterval{2, 1}, false},
	}
	for _, c := range cases {
		err := c.iv.Validate()
		if c.ok && err != nil {
			t.Errorf("%+v: %v", c.iv, err)
		}
		if !c.ok && err == nil {
			t.Errorf("%+v: want error", c.iv)
		}
	}
}

func TestAFTIntervalFromScalarLabel(t *testing.T) {
	ev := AFTIntervalFromScalarLabel(2.5)
	if ev.Lower != 2.5 || ev.Upper != 2.5 {
		t.Fatalf("event %+v", ev)
	}
	rc := AFTIntervalFromScalarLabel(-3)
	if rc.Lower != 3 || !math.IsInf(rc.Upper, 1) {
		t.Fatalf("censor %+v", rc)
	}
}

func TestNewAFTDense(t *testing.T) {
	vals := []float64{1, 0, 0, 1, 1, 1}
	labels := []float64{1, -2, 3}
	d, err := NewDense(vals, 3, 2, labels, nil)
	if err != nil {
		t.Fatal(err)
	}
	ivs := []AFTInterval{
		{1, 1},
		{2, math.Inf(1)},
		{0, 4},
	}
	aft, err := NewAFTDense(d, ivs)
	if err != nil {
		t.Fatal(err)
	}
	got, ok := AFTIntervalsOf(aft)
	if !ok || len(got) != 3 {
		t.Fatalf("intervals %v ok=%v", got, ok)
	}
}
