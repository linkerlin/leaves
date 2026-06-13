package io

import (
	"encoding/json"
	"fmt"
	"io"
	"os"

	ubjson "github.com/toitware/ubjson"
)

// ParseXGBoostUBJSON 从 reader 解析 XGBoost 3.x UBJSON 模型。
func ParseXGBoostUBJSON(r io.Reader) (*XGBoostLoadResult, error) {
	data, err := io.ReadAll(r)
	if err != nil {
		return nil, err
	}
	return parseXGBoostUBJSONBytes(data)
}

// ParseXGBoostUBJSONFile 从文件解析 XGBoost UBJSON 模型。
func ParseXGBoostUBJSONFile(filename string) (*XGBoostLoadResult, error) {
	data, err := os.ReadFile(filename)
	if err != nil {
		return nil, err
	}
	return parseXGBoostUBJSONBytes(data)
}

func parseXGBoostUBJSONBytes(data []byte) (*XGBoostLoadResult, error) {
	var root map[string]interface{}
	if err := ubjson.Unmarshal(data, &root); err != nil {
		return nil, fmt.Errorf("invalid xgboost ubjson: %w", err)
	}
	if root["learner"] == nil {
		return nil, fmt.Errorf("missing learner field")
	}
	jsonData, err := json.Marshal(root)
	if err != nil {
		return nil, fmt.Errorf("ubjson to json: %w", err)
	}
	return parseXGBoostJSONBytes(jsonData)
}
