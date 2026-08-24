package io

import (
	"path/filepath"
	"testing"
)

func TestLoadFromFileWithoutLegacyRegister(t *testing.T) {
	oldL, oldB := registeredLoader, registeredBuilder
	registeredLoader, registeredBuilder = nil, nil
	t.Cleanup(func() {
		registeredLoader, registeredBuilder = oldL, oldB
	})

	xgb := filepath.Join("..", "testdata", "xgboost_smoke.json")
	m, err := LoadFromFile(xgb, &LoadOptions{LoadTransformation: false, Backend: BackendNative})
	if err != nil {
		t.Fatalf("xgb json without register: %v", err)
	}
	defer m.Close()
	if m.NFeatures() <= 0 {
		t.Fatalf("nfeatures=%d", m.NFeatures())
	}

	lgb := filepath.Join("..", "testdata", "lg_breast_cancer.txt")
	lm, err := LoadFromFile(lgb, &LoadOptions{LoadTransformation: false, Backend: BackendNative})
	if err != nil {
		t.Fatalf("lgb text without register: %v", err)
	}
	defer lm.Close()
	if lm.NFeatures() <= 0 {
		t.Fatalf("lgb nfeatures=%d", lm.NFeatures())
	}
}

func TestLoadSklearnWithoutRegister(t *testing.T) {
	oldL, oldB := registeredLoader, registeredBuilder
	registeredLoader, registeredBuilder = nil, nil
	t.Cleanup(func() {
		registeredLoader, registeredBuilder = oldL, oldB
	})
	path := filepath.Join("..", "testdata", "sk_gradient_boosting_classifier.model")
	m, err := LoadFromFile(path, &LoadOptions{AutoTransform: false, Backend: BackendNative})
	if err != nil {
		t.Fatalf("sklearn without register: %v", err)
	}
	defer m.Close()
	if m.NFeatures() <= 0 {
		t.Fatalf("nfeatures=%d", m.NFeatures())
	}
}
