package train

import (
	"log"

	"github.com/dmitryikh/leaves/data"
	"github.com/dmitryikh/leaves/tree"
	"github.com/dmitryikh/leaves/treebuilder"
)

// ResolveTrainTreeMethod 解析训练建树算法；auto/gpu_hist 在 hist 路径上启用 GPU 增益扫描尝试。
func ResolveTrainTreeMethod(requested string, nRow int) (resolved string, useGPUHist bool) {
	switch requested {
	case "", treebuilder.MethodAuto:
		resolved = treebuilder.ResolveMethod(treebuilder.MethodAuto, nRow)
		useGPUHist = resolved != treebuilder.MethodExact
	case treebuilder.MethodGPUHist:
		resolved = treebuilder.MethodGPUHist
		useGPUHist = true
	case treebuilder.MethodExact:
		resolved = treebuilder.MethodExact
		useGPUHist = false
	default:
		resolved = treebuilder.ResolveMethod(requested, nRow)
		useGPUHist = resolved == treebuilder.MethodHist || resolved == treebuilder.MethodGPUHist
	}
	return resolved, useGPUHist
}

// ResolveAccelMode 解析训练加速模式；cfg 非空优先，否则读 LEAVES_TRAIN_ACCEL。
func ResolveAccelMode(cfgMode string) string {
	if cfgMode != "" {
		return cfgMode
	}
	return treebuilder.AccelModeFromEnv()
}

func usesHistTreeMethod(method string) bool {
	switch method {
	case treebuilder.MethodHist, treebuilder.MethodGPUHist:
		return true
	default:
		return false
	}
}

func (l *Learner) treebuilderCfg(dm data.Matrix) treebuilder.Config {
	cfg := treebuilder.Config{
		MaxDepth:      l.cfg.MaxDepth,
		MinHessian:    l.cfg.MinHessian,
		Lambda:        l.cfg.Lambda,
		Gamma:         l.cfg.Gamma,
		LearningRate:  l.cfg.LearningRate,
		MaxBin:        l.cfg.MaxBin,
		NumThreads:    l.cfg.NumThreads,
		UseGPUHist:    l.useGPUHist,
		AccelMode:     ResolveAccelMode(l.cfg.AccelMode),
		HistBinPolicy: l.cfg.HistBinPolicy,
	}
	method := l.resolvedTreeMethod
	if method == "" {
		method = l.cfg.TreeMethod
	}
	if usesHistTreeMethod(method) || (method == treebuilder.MethodAuto && l.useGPUHist) {
		if cfg.HistBinPolicy == "" {
			cfg.HistBinPolicy = treebuilder.HistBinGlobal
		}
		if cfg.HistBinPolicy == treebuilder.HistBinGlobal {
			cfg.GlobalBins = treebuilder.BuildGlobalHistBins(dm, l.cfg.MaxBin, nil)
		}
	}
	return cfg
}

func (l *Learner) beginTrainAccel(dm data.Matrix) {
	if l.accelLogged {
		return
	}
	requested := l.cfg.TreeMethod
	if requested == "" {
		requested = treebuilder.MethodAuto
	}
	resolved, useGPU := ResolveTrainTreeMethod(l.cfg.TreeMethod, dm.NumRow())
	l.resolvedTreeMethod = resolved
	l.useGPUHist = useGPU
	accelMode := ResolveAccelMode(l.cfg.AccelMode)
	treebuilder.ResetAccelStats()
	treebuilder.LogTrainAccelStart(
		requested,
		resolved,
		accelMode,
		useGPU,
		dm.NumRow(),
		tree.BornWebGPUAvailable(),
		treebuilder.BornHistAvailable(),
	)
	l.accelLogged = true
}

func (l *Learner) endTrainAccel() {
	if !l.accelLogged {
		return
	}
	treebuilder.LogTrainAccelEnd()
	if l.marginPredictGPU+l.marginPredictCPU > 0 {
		log.Printf(
			"[leaves/train] accel margin: gpu_calls=%d cpu_calls=%d",
			l.marginPredictGPU, l.marginPredictCPU,
		)
	}
	l.accelLogged = false
}
