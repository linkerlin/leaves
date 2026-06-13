package io

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/dmitryikh/leaves/model"
)

// LoaderFunc 由根包注册的遗留模型加载函数。
// 返回 *leaves.Ensemble，以 interface{} 避免 io→leaves 循环依赖。
type LoaderFunc func(filename string, opts *LoadOptions) (interface{}, error)

// EngineBuilder 从遗留 Ensemble 构建 model.Ensemble。
type EngineBuilder func(legacy interface{}, opts *LoadOptions) (*model.Ensemble, error)

var (
	registeredLoader  LoaderFunc
	registeredBuilder EngineBuilder
)

// RegisterLegacyLoader 注册根包加载器（在 leaves.init 中调用）。
func RegisterLegacyLoader(loader LoaderFunc, builder EngineBuilder) {
	registeredLoader = loader
	registeredBuilder = builder
}

// DetectFormat 根据文件内容/扩展名检测模型格式。
func DetectFormat(filename string) (Format, error) {
	ext := strings.ToLower(filepath.Ext(filename))
	switch ext {
	case ".json":
		return detectJSONFormat(filename)
	case ".ubj":
		return FormatXGBoostUBJSON, nil
	case ".model", ".bin":
		return detectBinaryFormat(filename)
	case ".txt":
		return FormatLightGBM, nil
	default:
		return detectBinaryFormat(filename)
	}
}

func detectJSONFormat(filename string) (Format, error) {
	data, err := os.ReadFile(filename)
	if err != nil {
		return FormatUnknown, err
	}
	var probe struct {
		LeavesVersion int             `json:"leaves_version"`
		TreeInfo      json.RawMessage `json:"tree_info"`
		Learner       json.RawMessage `json:"learner"`
	}
	if err := json.Unmarshal(data, &probe); err != nil {
		return FormatUnknown, fmt.Errorf("invalid json: %w", err)
	}
	if probe.LeavesVersion > 0 {
		return FormatLeavesJSON, nil
	}
	if probe.Learner != nil {
		return FormatXGBoostJSON, nil
	}
	if probe.TreeInfo != nil {
		return FormatLightGBMJSON, nil
	}
	return FormatLightGBMJSON, nil
}

func detectBinaryFormat(filename string) (Format, error) {
	f, err := os.Open(filename)
	if err != nil {
		return FormatUnknown, err
	}
	defer f.Close()

	// XGBoost 旧二进制魔数 "binf"
	buf := make([]byte, 4)
	n, _ := f.Read(buf)
	if n >= 2 && buf[0] == '{' && buf[1] != '"' {
		return FormatXGBoostUBJSON, nil
	}
	if n == 4 && string(buf) == "binf" {
		return FormatXGBoost, nil
	}

	// pickle 魔数
	if n >= 2 && buf[0] == 0x80 && buf[1] >= 0x02 {
		return FormatSklearn, nil
	}

	// LightGBM 文本：tree= 或 version=
	f.Seek(0, 0)
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := scanner.Text()
		if strings.HasPrefix(line, "tree=") || strings.TrimSpace(line) == "tree" || strings.HasPrefix(line, "version=") {
			return FormatLightGBM, nil
		}
		if strings.TrimSpace(line) != "" {
			break
		}
	}

	// 默认尝试 XGBoost
	if n == 4 && bytes.Equal(buf, []byte{0x00, 0x05, 0x5f, 0x00}) {
		return FormatXGBoost, nil
	}
	return FormatXGBoost, nil
}

// LoadFromFile 从文件自动检测格式并加载模型。
func LoadFromFile(filename string, opts *LoadOptions) (*model.Ensemble, error) {
	if opts == nil {
		opts = DefaultLoadOptions()
	}
	if registeredLoader == nil || registeredBuilder == nil {
		return nil, fmt.Errorf("io loader not registered: import github.com/dmitryikh/leaves to enable")
	}

	legacy, err := registeredLoader(filename, opts)
	if err != nil {
		return nil, err
	}
	if legacy == nil {
		return nil, fmt.Errorf("loader returned nil model for %s", filename)
	}
	return registeredBuilder(legacy, opts)
}
