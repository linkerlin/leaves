// Package model 提供基于 tree.Engine 的统一模型封装。
// 这是 leaves v1.0 的推荐 API——使用 tree.Engine 作为可插拔后端。
// 根包中的旧 Ensemble 类型在 v1.0 之前仍可用，通过类型别名保持兼容。
package model

import (
	"fmt"
	"math"
	"runtime"
	"sync"

	"github.com/dmitryikh/leaves/predict"
)

// BatchSize 并行批预测的批次大小。
const BatchSize = 16

// Ensemble 基于 predict.Engine 的统一模型封装。
// 提供 PredictSingle / Predict / PredictDense / PredictCSR 等公开方法。
type Ensemble struct {
	engine predict.Engine
}

// NewEnsemble 用 predict.Engine 创建 Ensemble。
func NewEnsemble(engine predict.Engine) *Ensemble {
	return &Ensemble{engine: engine}
}

// ---- 预测 API ----

// PredictSingle 单样本单输出预测。
func (e *Ensemble) PredictSingle(fvals []float64, nEstimators int) float64 {
	return e.engine.PredictSingle(fvals, nEstimators)
}

// Predict 单样本预测（支持多输出）。
func (e *Ensemble) Predict(fvals []float64, nEstimators int, predictions []float64) error {
	return e.engine.Predict(fvals, nEstimators, predictions)
}

// PredictDense 稠密矩阵批量预测。
// nThreads 控制并发数（0=单线程，>GOMAXPROCS=自动调整）。
func (e *Ensemble) PredictDense(
	vals []float64, nrows int, ncols int,
	predictions []float64,
	nEstimators int,
	nThreads int,
) error {
	nRows := nrows
	if len(predictions) < e.NOutputGroups()*nRows {
		return fmt.Errorf("predictions slice too short (need at least %d)", e.NOutputGroups()*nRows)
	}
	if ncols == 0 || e.NFeatures() > ncols {
		return fmt.Errorf("incorrect number of columns")
	}

	if nRows <= BatchSize || nThreads <= 1 {
		return e.engine.PredictDense(vals, nrows, ncols, predictions, nEstimators)
	}

	// 多线程批预测
	if nThreads > runtime.GOMAXPROCS(0) {
		nThreads = runtime.GOMAXPROCS(0)
	}
	nBatches := int(math.Ceil(float64(nRows) / BatchSize))
	if nThreads > nBatches {
		nThreads = nBatches
	}

	tasks := make(chan int)
	wg := sync.WaitGroup{}
	for i := 0; i < nThreads; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for startIdx := range tasks {
				endIdx := startIdx + BatchSize
				if endIdx > nRows {
					endIdx = nRows
				}
				for i := startIdx; i < endIdx; i++ {
					fvals := vals[i*ncols : (i+1)*ncols]
					e.engine.Predict(fvals, nEstimators, predictions[i*e.NOutputGroups():(i+1)*e.NOutputGroups()])
				}
			}
		}()
	}

	for i := 0; i < nBatches; i++ {
		tasks <- i * BatchSize
	}
	close(tasks)
	wg.Wait()
	return nil
}

// PredictCSR CSR 稀疏矩阵批量预测。
// 始终委托引擎 CSR 实现（保证 XGBoost NaN 缺失语义正确）。
func (e *Ensemble) PredictCSR(
	indptr []int, cols []int, vals []float64,
	predictions []float64,
	nEstimators int,
	nThreads int,
) error {
	nRows := len(indptr) - 1
	if len(predictions) < e.NOutputGroups()*nRows {
		return fmt.Errorf("predictions slice too short (need at least %d)", e.NOutputGroups()*nRows)
	}
	_ = nThreads // 线程池在引擎层扩展；当前保证语义正确优先。
	return e.engine.PredictCSR(indptr, cols, vals, predictions, nEstimators)
}

// ---- 元数据查询 ----

// NEstimators 每组估计器数量。
func (e *Ensemble) NEstimators() int       { return e.engine.NEstimators() }

// NRawOutputGroups 原始输出维度。
func (e *Ensemble) NRawOutputGroups() int   { return e.engine.NRawOutputGroups() }

// NOutputGroups 变换后输出维度。
func (e *Ensemble) NOutputGroups() int      { return e.engine.NOutputGroups() }

// NFeatures 输入特征数。
func (e *Ensemble) NFeatures() int          { return e.engine.NFeatures() }

// NLeaves 每棵树叶子数。
func (e *Ensemble) NLeaves() []int          { return e.engine.NLeaves() }

// Name 模型名称。
func (e *Ensemble) Name() string            { return e.engine.Name() }

// Engine 返回底层 predict.Engine（高级用法）。
func (e *Ensemble) Engine() predict.Engine { return e.engine }

// ReplaceEngine 替换底层引擎并关闭旧引擎（热更新内部使用）。
func (e *Ensemble) ReplaceEngine(next predict.Engine) error {
	if e == nil {
		return fmt.Errorf("model: nil ensemble")
	}
	old := e.engine
	e.engine = next
	if old != nil {
		if err := old.Close(); err != nil {
			return fmt.Errorf("model: close old engine: %w", err)
		}
	}
	return nil
}

// DetachEngine 取出引擎引用并将 Ensemble 置空（避免重复 Close）。
func (e *Ensemble) DetachEngine() predict.Engine {
	if e == nil {
		return nil
	}
	eng := e.engine
	e.engine = nil
	return eng
}

// Close 释放资源。
func (e *Ensemble) Close() error {
	if e == nil || e.engine == nil {
		return nil
	}
	return e.engine.Close()
}
