package io_test

import (
	"bytes"
	"encoding/json"
	"path/filepath"
	"testing"

	"github.com/linkerlin/leaves/io"
)

func TestExportXGBoostJSONFeatureMeta(t *testing.T) {
	path := filepath.Join("..", "testdata", "xgboost_smoke.json")
	result, err := io.ParseXGBoostJSONFile(path)
	if err != nil {
		t.Fatal(err)
	}
	ir := result.IR
	ir.FeatureNames = []string{"a", "b", "c", "d", "e", "f", "g", "h"}
	ir.FeatureTypes = []string{"float", "float", "float", "float", "float", "float", "float", "c"}

	var buf bytes.Buffer
	if err := io.ExportXGBoostJSON(&buf, ir, result.Objective); err != nil {
		t.Fatal(err)
	}
	var doc map[string]interface{}
	if err := json.Unmarshal(buf.Bytes(), &doc); err != nil {
		t.Fatal(err)
	}
	learner := doc["learner"].(map[string]interface{})
	names := learner["feature_names"].([]interface{})
	types := learner["feature_types"].([]interface{})
	if len(names) != 8 || names[0] != "a" {
		t.Fatalf("feature_names: %v", names)
	}
	if len(types) != 8 || types[7] != "c" {
		t.Fatalf("feature_types: %v", types)
	}
}
