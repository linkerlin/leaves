package leaves

import (
	"github.com/linkerlin/leaves/model"
)

// ensembleCore 返回底层 booster 实现（解开 Ensemble 包装层）。
func (e *Ensemble) ensembleCore() ensembleBaseInterface {
	base := e.ensembleBaseInterface
	for {
		wrapped, ok := base.(*Ensemble)
		if !ok {
			return base
		}
		base = wrapped.ensembleBaseInterface
	}
}

func (e *Ensemble) modelProxy() (*model.Ensemble, error) {
	e.proxyOnce.Do(func() {
		opts := e.engineOpts
		if opts == nil {
			opts = DefaultEngineOptions()
		}
		// 用解开后的 core + 当前 transform 构建 IR。
		core := &Ensemble{ensembleBaseInterface: e.ensembleCore(), transform: e.transform}
		e.proxy, e.proxyErr = NewModelEnsemble(core, opts)
	})
	return e.proxy, e.proxyErr
}

func (e *Ensemble) viaProxy() (*model.Ensemble, bool) {
	proxy, err := e.modelProxy()
	return proxy, err == nil && proxy != nil
}

// WithEngineOptions 返回共享 booster、指定引擎选项的新 Ensemble 视图。
func (e *Ensemble) WithEngineOptions(opts *EngineOptions) *Ensemble {
	return &Ensemble{
		ensembleBaseInterface: e.ensembleCore(),
		transform:             e.transform,
		engineOpts:            opts,
	}
}
