package model

import "fmt"

// ReloadLoader 从 path 加载新 Ensemble（由 leaves 根包注册，避免 model↔io 循环依赖）。
type ReloadLoader func(path string, opts any) (*Ensemble, error)

var registeredReloadLoader ReloadLoader

// RegisterReloadLoader 注册热加载函数（在 leaves.init 中调用）。
func RegisterReloadLoader(fn ReloadLoader) {
	registeredReloadLoader = fn
}

// Reload 从 path 热加载模型并替换当前引擎；opts 类型由注册的 loader 定义（通常为 *io.LoadOptions）。
// 须 import github.com/linkerlin/leaves 以注册 loader。
func (e *Ensemble) Reload(path string, opts any) error {
	if e == nil {
		return fmt.Errorf("model: nil ensemble")
	}
	if registeredReloadLoader == nil {
		return fmt.Errorf("model: reload loader not registered: import github.com/linkerlin/leaves")
	}
	next, err := registeredReloadLoader(path, opts)
	if err != nil {
		return err
	}
	if next == nil {
		return fmt.Errorf("model: loader returned nil ensemble for %s", path)
	}
	eng := next.DetachEngine()
	if eng == nil {
		return fmt.Errorf("model: loader returned empty engine for %s", path)
	}
	return e.ReplaceEngine(eng)
}
