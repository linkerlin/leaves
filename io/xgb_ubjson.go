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

func parseXGBoostUBJSONBytes(data []byte) (res *XGBoostLoadResult, err error) {
	// ponytail: toitware/ubjson 对畸形输入可能 panic（负长度切片等），无上游版本可升；
	// 信任边界统一在此兜底转错误，上游修复后可移除。
	defer func() {
		if r := recover(); r != nil {
			res, err = nil, fmt.Errorf("invalid xgboost ubjson (panic in ubjson decoder): %v", r)
		}
	}()
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
