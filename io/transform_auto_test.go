package io_test

import (
	"testing"

	"github.com/linkerlin/leaves/io"
)

func TestDefaultLoadOptionsAutoTransform(t *testing.T) {
	opts := io.DefaultLoadOptions()
	if !opts.AutoTransform {
		t.Fatal("DefaultLoadOptions should enable AutoTransform")
	}
	if io.ResolveLoadTransformation(opts, "binary:logistic") != true {
		t.Fatal("expected transform for binary:logistic with defaults")
	}
	if io.ResolveLoadTransformation(opts, "reg:squarederror") {
		t.Fatal("reg:squarederror should stay raw with AutoTransform only")
	}
}

func TestResolveLoadTransformation(t *testing.T) {
	opts := &io.LoadOptions{AutoTransform: true}
	if !io.ResolveLoadTransformation(opts, "binary:logistic") {
		t.Fatal("expected auto transform for binary:logistic")
	}
	if io.ResolveLoadTransformation(opts, "reg:squarederror") {
		t.Fatal("unexpected transform for reg:squarederror")
	}
	opts.LoadTransformation = true
	if !io.ResolveLoadTransformation(opts, "reg:squarederror") {
		t.Fatal("LoadTransformation should force true")
	}
}

func TestObjectiveNeedsTransform(t *testing.T) {
	cases := []struct {
		obj  string
		want bool
	}{
		{"binary:logistic", true},
		{"multi:softprob", true},
		{"reg:gamma", true},
		{"reg:squarederror", false},
		{"rank:ndcg", false},
	}
	for _, c := range cases {
		if got := io.ObjectiveNeedsTransform(c.obj); got != c.want {
			t.Fatalf("ObjectiveNeedsTransform(%q)=%v want %v", c.obj, got, c.want)
		}
	}
}
