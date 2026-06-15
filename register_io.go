package leaves

import (
	"os"
	"strings"

	"github.com/dmitryikh/leaves/io"
	"github.com/dmitryikh/leaves/model"
	"github.com/dmitryikh/leaves/tree"
)

func init() {
	io.RegisterLegacyLoader(legacyLoadFromFile, legacyBuildModelEnsemble)
	model.RegisterReloadLoader(func(path string, opts any) (*model.Ensemble, error) {
		lo, _ := opts.(*io.LoadOptions)
		return io.LoadFromFile(path, lo)
	})
}

func legacyLoadFromFile(filename string, opts *io.LoadOptions) (interface{}, error) {
	format, err := io.DetectFormat(filename)
	if err != nil {
		if result, tryErr := io.ParseXGBoostBinaryFile(filename); tryErr == nil {
			return result, nil
		}
		return nil, err
	}

	loadTransform := opts != nil && opts.LoadTransformation

	switch format {
	case io.FormatLightGBM:
		return LGEnsembleFromFile(filename, loadTransform)
	case io.FormatLightGBMJSON:
		f, err := os.Open(filename)
		if err != nil {
			return nil, err
		}
		defer f.Close()
		return LGEnsembleFromJSON(f, loadTransform)
	case io.FormatXGBoost:
		return io.ParseXGBoostBinaryFile(filename)
	case io.FormatXGBoostJSON:
		return io.ParseXGBoostJSONFile(filename)
	case io.FormatXGBoostUBJSON:
		return io.ParseXGBoostUBJSONFile(filename)
	case io.FormatLeavesJSON:
		return io.LoadLeavesJSONFile(filename)
	case io.FormatSklearn:
		return SKEnsembleFromFile(filename, loadTransform)
	default:
		if strings.HasSuffix(strings.ToLower(filename), ".json") {
			f, err := os.Open(filename)
			if err != nil {
				return nil, err
			}
			defer f.Close()
			return LGEnsembleFromJSON(f, loadTransform)
		}
		return nil, io.ErrFormatNotImplemented("unknown format")
	}
}

func legacyBuildModelEnsemble(legacy interface{}, opts *io.LoadOptions) (*model.Ensemble, error) {
	if result, ok := legacy.(*io.XGBoostLoadResult); ok {
		return buildFromXGBJSONResult(result, opts)
	}
	if lr, ok := legacy.(*io.LeavesLoadResult); ok {
		return buildFromLeavesResult(lr, opts)
	}
	e, ok := legacy.(*Ensemble)
	if !ok {
		return nil, io.ErrFormatNotImplemented("unexpected legacy model type")
	}
	engineOpts := DefaultEngineOptions()
	if opts != nil {
		engineOpts.Backend = opts.Backend
		engineOpts.Workload = opts.Workload
	}
	return NewModelEnsemble(e, engineOpts)
}

func buildFromXGBJSONResult(result *io.XGBoostLoadResult, opts *io.LoadOptions) (*model.Ensemble, error) {
	loadTransform := io.ResolveLoadTransformation(opts, result.Objective)
	backend := tree.BackendNative
	hint := tree.DefaultWorkloadHint()
	if opts != nil {
		backend = opts.Backend
		hint = opts.Workload
	}
	outType, transform := io.ObjectiveToTransform(result.Objective, loadTransform)
	result.IR.NOutputGroups = io.NOutputGroupsForTransform(result.IR.NRawOutputGroups, outType)
	return model.NewEnsembleFromIRWithHint(result.IR, transform, outType, backend, hint)
}

func buildFromLeavesResult(result *io.LeavesLoadResult, opts *io.LoadOptions) (*model.Ensemble, error) {
	loadTransform := io.ResolveLoadTransformation(opts, result.Objective)
	backend := tree.BackendNative
	hint := tree.DefaultWorkloadHint()
	if opts != nil {
		backend = opts.Backend
		hint = opts.Workload
	}
	outType, transform := io.ObjectiveToTransform(result.Objective, loadTransform)
	result.IR.NOutputGroups = io.NOutputGroupsForTransform(result.IR.NRawOutputGroups, outType)
	return model.NewEnsembleFromIRWithHint(result.IR, transform, outType, backend, hint)
}
