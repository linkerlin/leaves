package modelhost

import (
	"fmt"
	"sync"

	"github.com/linkerlin/leaves/v2"
	"github.com/linkerlin/leaves/v2/model"
)

// Host 线程安全的模型持有者；支持热替换（Reload）。
type Host struct {
	mu    sync.RWMutex
	ens   *model.Ensemble
	path  string
	nFeat int
	nOut  int
}

// Load 从 path 加载模型。
func Load(path string) (*Host, error) {
	ens, err := leaves.LoadFromFile(path, leaves.DefaultLoadOptions())
	if err != nil {
		return nil, fmt.Errorf("load model %s: %w", path, err)
	}
	return &Host{
		ens:   ens,
		path:  path,
		nFeat: ens.NFeatures(),
		nOut:  ens.NOutputGroups(),
	}, nil
}

// Path 当前模型路径。
func (h *Host) Path() string {
	h.mu.RLock()
	defer h.mu.RUnlock()
	return h.path
}

// Meta 特征维 / 输出维。
func (h *Host) Meta() (nFeat, nOut int) {
	h.mu.RLock()
	defer h.mu.RUnlock()
	return h.nFeat, h.nOut
}

// Ready 是否已加载。
func (h *Host) Ready() bool {
	h.mu.RLock()
	defer h.mu.RUnlock()
	return h.ens != nil
}

// Reload 从新路径加载并原子替换；失败保留旧模型。
func (h *Host) Reload(path string) error {
	ens, err := leaves.LoadFromFile(path, leaves.DefaultLoadOptions())
	if err != nil {
		return fmt.Errorf("reload %s: %w", path, err)
	}
	h.mu.Lock()
	old := h.ens
	h.ens = ens
	h.path = path
	h.nFeat = ens.NFeatures()
	h.nOut = ens.NOutputGroups()
	h.mu.Unlock()
	if old != nil {
		_ = old.Close()
	}
	return nil
}

// PredictDense 批预测；调用方保证 vals 长度 nrows*ncols。
func (h *Host) PredictDense(vals []float64, nrows, ncols int, out []float64) error {
	h.mu.RLock()
	ens := h.ens
	h.mu.RUnlock()
	if ens == nil {
		return fmt.Errorf("model not loaded")
	}
	return ens.PredictDense(vals, nrows, ncols, out, 0, 0)
}

// PredictSingle 单条。
func (h *Host) PredictSingle(x []float64) float64 {
	h.mu.RLock()
	ens := h.ens
	h.mu.RUnlock()
	if ens == nil {
		return 0
	}
	return ens.PredictSingle(x, 0)
}

// Close 释放模型。
func (h *Host) Close() error {
	h.mu.Lock()
	defer h.mu.Unlock()
	if h.ens == nil {
		return nil
	}
	err := h.ens.Close()
	h.ens = nil
	return err
}
