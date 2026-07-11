package io

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/linkerlin/leaves/internal/xgbin"
	"github.com/linkerlin/leaves/model"
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
	case ".pkl", ".joblib":
		return FormatSklearn, nil
	case ".onnx":
		// 占位：明确识别，避免落入 unrecognized 杂讯。
		return FormatONNX, nil
	case ".model", ".bin":
		return detectBinaryFormat(filename)
	case ".txt":
		return detectTextFormat(filename)
	default:
		return detectBinaryFormat(filename)
	}
}

func detectTextFormat(filename string) (Format, error) {
	f, err := os.Open(filename)
	if err != nil {
		return FormatUnknown, err
	}
	defer f.Close()
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		if strings.HasPrefix(line, "tree=") || strings.TrimSpace(line) == "tree" || strings.HasPrefix(line, "version=") {
			return FormatLightGBM, nil
		}
		// 数值 TSV/CSV 训练数据误用 .txt：给出明确提示
		if looksLikeNumericTableLine(line) {
			return FormatUnknown, &LoadError{
				Path:   filename,
				Format: FormatUnknown,
				Level:  SupportUnsupported,
				Op:     "detect",
				Msg:    "looks like tabular training data, not a model file",
				Hint:   "训练数据请用 data.FromFile / leaves train --data；模型请用 LGB/XGB/leaves.json",
			}
		}
		break
	}
	return detectBinaryFormat(filename)
}

func looksLikeNumericTableLine(line string) bool {
	delim := ','
	if strings.Count(line, "\t") > strings.Count(line, ",") {
		delim = '\t'
	}
	var parts []string
	if delim == '\t' {
		parts = strings.Split(line, "\t")
	} else {
		parts = strings.Split(line, ",")
	}
	if len(parts) < 2 {
		return false
	}
	numeric := 0
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p == "" {
			return false
		}
		if _, err := strconv.ParseFloat(p, 64); err == nil {
			numeric++
		}
	}
	return numeric == len(parts)
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

	// pickle 魔数（protocol 2+）
	if n >= 2 && buf[0] == 0x80 && buf[1] >= 0x02 {
		return FormatSklearn, nil
	}
	// pickle protocol 0（文本 opcode，如 ccopy_reg）
	if n >= 4 && buf[0] == 'c' && string(buf[:4]) == "ccop" {
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

	// 经典 XGB 二进制无固定魔数：尝试解析 header（无效数据不得 panic）
	if probeXGBBinaryHeader(f) {
		return FormatXGBoost, nil
	}
	sup := SupportOf(FormatUnknown)
	return FormatUnknown, &LoadError{
		Path:   filename,
		Format: FormatUnknown,
		Level:  SupportUnsupported,
		Op:     "detect",
		Msg:    "unrecognized model format",
		Hint:   sup.Hint,
	}
}

func probeXGBBinaryHeader(f *os.File) bool {
	if _, err := f.Seek(0, 0); err != nil {
		return false
	}
	ok := false
	func() {
		defer func() {
			if recover() != nil {
				ok = false
			}
		}()
		if _, err := xgbin.ReadModelHeader(bufio.NewReader(f)); err == nil {
			ok = true
		}
	}()
	return ok
}

// LoadFromFile 从文件自动检测格式并加载模型。
// 失败时返回 *LoadError（含 support level 与 hint），便于 Agent/人类排障。
func LoadFromFile(filename string, opts *LoadOptions) (*model.Ensemble, error) {
	if opts == nil {
		opts = DefaultLoadOptions()
	}
	if registeredLoader == nil || registeredBuilder == nil {
		return nil, fmt.Errorf("io loader not registered: import github.com/linkerlin/leaves to enable")
	}

	// 先探测：ONNX 等占位格式直接可操作失败，不进入遗留 loader。
	format, derr := DetectFormat(filename)
	if derr != nil {
		return nil, wrapDetectError(filename, derr)
	}
	if format == FormatONNX {
		// LIB-10：TreeEnsembleRegressor 极小子集；失败带 experimental hint。
		return LoadONNX(filename, opts)
	}

	legacy, err := registeredLoader(filename, opts)
	if err != nil {
		// 探测与 loader 可能不一致时，以探测结果标注等级；未知则再探测一次。
		if format == FormatUnknown {
			if f2, e2 := DetectFormat(filename); e2 == nil {
				format = f2
			}
		}
		return nil, wrapLoadError(filename, format, err)
	}
	if legacy == nil {
		return nil, wrapLoadError(filename, format, fmt.Errorf("loader returned nil model"))
	}
	ens, err := registeredBuilder(legacy, opts)
	if err != nil {
		return nil, wrapLoadError(filename, format, err)
	}
	return ens, nil
}
