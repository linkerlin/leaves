package leaves

// finalizeLegacyEnsemble 为根包遗留加载路径启用 model.Ensemble 委托。
func finalizeLegacyEnsemble(e *Ensemble) *Ensemble {
	if e == nil {
		return nil
	}
	if e.engineOpts == nil {
		e.engineOpts = DefaultEngineOptions()
	}
	return e
}
