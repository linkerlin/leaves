package treebuilder

import "sync"

type gpuHistJob struct {
	feats          []int
	idx            []int
	grad, hess     []float64
	cfg            Config
	done           chan map[int]gpuHistResult
}

var (
	gpuHistQueueOnce sync.Once
	gpuHistQueue     chan gpuHistJob
)

func startGPUHistWorker() {
	gpuHistQueueOnce.Do(func() {
		gpuHistQueue = make(chan gpuHistJob, 32)
		go func() {
			for job := range gpuHistQueue {
				var result map[int]gpuHistResult
				if gpuHistBatchEnabled(job.cfg) && len(job.feats) > 0 {
					result = batchAccumulateHistWebGPU(job.feats, job.idx, job.grad, job.hess, job.cfg)
				}
				job.done <- result
			}
		}()
	})
}

// enqueueGPUHistBatch 将 GPU 直方图 batch 提交到单 worker 队列；调用方可并行做 CPU hist。
func enqueueGPUHistBatch(
	feats []int,
	idx []int,
	grad, hess []float64,
	cfg Config,
) <-chan map[int]gpuHistResult {
	ch := make(chan map[int]gpuHistResult, 1)
	if len(feats) == 0 || !gpuHistBatchEnabled(cfg) {
		ch <- nil
		return ch
	}
	startGPUHistWorker()
	gpuHistQueue <- gpuHistJob{
		feats: feats,
		idx:   idx,
		grad:  grad,
		hess:  hess,
		cfg:   cfg,
		done:  ch,
	}
	return ch
}
