package train_test

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"testing"
	"time"

	"github.com/dmitryikh/leaves/data"
	"github.com/dmitryikh/leaves/train"
	"github.com/dmitryikh/leaves/tree"
	"github.com/dmitryikh/leaves/treebuilder"
)

type accelBenchCase struct {
	name       string
	accelMode  string
	treeMethod string
	numThreads int
}

func accelBenchEnvInt(name string, def int) int {
	v := os.Getenv(name)
	if v == "" {
		return def
	}
	n, err := strconv.Atoi(v)
	if err != nil || n <= 0 {
		return def
	}
	return n
}

func skipUnlessAccelBench(t *testing.T) {
	t.Helper()
	if testing.Short() {
		t.Skip("accel benchmark skipped in -short mode")
	}
	switch os.Getenv("LEAVES_BENCH") {
	case "1", "true", "yes":
	default:
		t.Skip("accel benchmark skipped (set LEAVES_BENCH=1; see README §训练加速 benchmark)")
	}
}

func filterAccelBenchCases(cases []accelBenchCase) []accelBenchCase {
	only := os.Getenv("LEAVES_BENCH_ONLY")
	if only == "" {
		return cases
	}
	filtered := cases[:0]
	for _, c := range cases {
		if c.name == only {
			filtered = append(filtered, c)
		}
	}
	if len(filtered) == 0 {
		return nil
	}
	return filtered
}

func gainGPUShare(s treebuilder.AccelStats) float64 {
	total := s.GainScanWebGPU + s.GainScanBornCPU + s.GainScanPureCPU
	if total == 0 {
		return 0
	}
	return float64(s.GainScanWebGPU) / float64(total)
}

func histGPUShare(s treebuilder.AccelStats) float64 {
	total := s.HistBuildCPU + s.HistBuildWebGPU
	if total == 0 {
		return 0
	}
	return float64(s.HistBuildWebGPU) / float64(total)
}

func runAccelBenchCase(
	t *testing.T,
	c accelBenchCase,
	cfg train.Config,
	dm data.Matrix,
	quality func(t *testing.T, learner *train.Learner, dm data.Matrix) float64,
) {
	learner, err := train.NewLearner(cfg)
	if err != nil {
		t.Fatalf("%s: NewLearner: %v", c.name, err)
	}
	start := time.Now()
	fitErr := func() (err error) {
		defer func() {
			if r := recover(); r != nil {
				err = fmt.Errorf("panic: %v", r)
			}
		}()
		return learner.Fit(dm)
	}()
	if fitErr != nil {
		t.Logf("%s\tFAIL\t—\t— (%v)", c.name, fitErr)
		return
	}
	elapsed := time.Since(start)
	stats := treebuilder.SnapshotAccelStats()
	q := quality(t, learner, dm)
	t.Logf("%s\t%.1f\t%.6f\thist_gpu=%.1f%%\tgain_gpu=%.1f%%\t%s",
		c.name, elapsed.Seconds(), q, 100*histGPUShare(stats), 100*gainGPUShare(stats), treebuilder.AccelSummary())
	fmt.Fprintf(os.Stderr,
		"[accel-bench] %s elapsed=%.1fs quality=%.6f hist_gpu=%.1f%% gain_gpu=%.1f%%\n",
		c.name, elapsed.Seconds(), q, 100*histGPUShare(stats), 100*gainGPUShare(stats))
}

func synthBenchDense(rows, cols int) data.Matrix {
	vals := make([]float64, rows*cols)
	labels := make([]float64, rows)
	for i := 0; i < rows; i++ {
		var y float64
		for j := 0; j < cols; j++ {
			v := float64(i)*0.0007 + float64(j)*0.11 + float64((i*j)%17)*0.03
			vals[i*cols+j] = v
			if j < 8 {
				y += v * float64(j+1) * 0.01
			}
		}
		labels[i] = y + float64(i%13)*0.05
	}
	dm, _ := data.NewDense(vals, rows, cols, labels, nil)
	return dm
}

// TestMSLTRTrainAccelBenchmark MSLR-WEB10K 子集训练加速路径对比（需 LEAVES_BENCH=1）。
func TestMSLTRTrainAccelBenchmark(t *testing.T) {
	skipUnlessAccelBench(t)
	bl := loadRankXGBBaseline(t, filepath.Join("..", "testdata", "rank_msltr_ndcg_xgb_baseline.json"))
	trainDM, err := data.LoadRankingTSV(filepath.Join("..", "testdata", "rank_msltr_train.tsv"), "\t")
	if err != nil {
		t.Fatalf("load train: %v", err)
	}
	testDM, err := data.LoadRankingTSV(filepath.Join("..", "testdata", "rank_msltr_test.tsv"), "\t")
	if err != nil {
		t.Fatalf("load test: %v", err)
	}

	cases := []accelBenchCase{
		{name: "cpu_hist", accelMode: train.AccelModeCPU, treeMethod: train.TreeMethodHist, numThreads: 4},
		{name: "auto_hist", accelMode: train.AccelModeAuto, treeMethod: train.TreeMethodHist, numThreads: 4},
		{name: "auto_smart", accelMode: train.AccelModeAuto, treeMethod: train.TreeMethodAuto, numThreads: 4},
	}
	if tree.BornWebGPUAvailable() {
		cases = append(cases,
			accelBenchCase{
				name: "webgpu_hist", accelMode: train.AccelModeWebGPU,
				treeMethod: train.TreeMethodGPUHist, numThreads: 4,
			},
		)
	}
	if filtered := filterAccelBenchCases(cases); filtered == nil {
		t.Fatalf("unknown LEAVES_BENCH_ONLY=%q", os.Getenv("LEAVES_BENCH_ONLY"))
	} else {
		cases = filtered
	}

	t.Logf("MSLTR accel benchmark: rows=%d rounds=%d webgpu=%v",
		trainDM.NumRow(), bl.NumRound, tree.BornWebGPUAvailable())
	t.Log("mode\tseconds\ttrain_ndcg\ttest_ndcg\thist_gpu%\taccel_summary")

	for _, c := range cases {
		cfg := train.Config{
			Objective:      bl.Objective,
			NumRound:       bl.NumRound,
			MaxDepth:       bl.MaxDepth,
			LearningRate:   bl.LearningRate,
			Lambda:         bl.Lambda,
			TreeMethod:     c.treeMethod,
			AccelMode:      c.accelMode,
			NumThreads:     c.numThreads,
			Seed:           int64(bl.Seed),
			NDCGK:          bl.NDCGK,
			EvalMetric:     bl.EvalMetric,
			LambdaRankNorm: true,
			HistBinPolicy:  "global",
		}
		learner, err := train.NewLearner(cfg)
		if err != nil {
			t.Fatal(err)
		}
		start := time.Now()
		fitErr := func() (err error) {
			defer func() {
				if r := recover(); r != nil {
					err = fmt.Errorf("panic: %v", r)
				}
			}()
			return learner.Fit(trainDM)
		}()
		if fitErr != nil {
			t.Logf("%s\tFAIL\t—\t—\t— (%v)", c.name, fitErr)
			continue
		}
		elapsed := time.Since(start)
		stats := treebuilder.SnapshotAccelStats()
		trainPreds := make([]float64, trainDM.NumRow())
		testPreds := make([]float64, testDM.NumRow())
		_ = learner.PredictMargins(trainDM, trainPreds)
		_ = learner.PredictMargins(testDM, testPreds)
		trainNDCG := ndcgOnMatrix(t, trainDM, trainPreds, bl.NDCGK)
		testNDCG := ndcgOnMatrix(t, testDM, testPreds, bl.NDCGK)
		t.Logf("%s\t%.1f\t%.4f\t%.4f\t%.1f%%\t%s",
			c.name, elapsed.Seconds(), trainNDCG, testNDCG, 100*histGPUShare(stats), treebuilder.AccelSummary())
		fmt.Fprintf(os.Stderr,
			"[msltr-bench] %s elapsed=%.1fs train_ndcg=%.4f test_ndcg=%.4f hist_gpu=%.1f%%\n",
			c.name, elapsed.Seconds(), trainNDCG, testNDCG, 100*histGPUShare(stats))
	}
}

// TestLargeDenseTrainAccelBenchmark 大规模稠密回归，验证 GPU hist 交叉点（需 LEAVES_BENCH=1）。
// 环境变量：LEAVES_BENCH_ROWS（默认 50000）、LEAVES_BENCH_COLS（64）、LEAVES_BENCH_ROUNDS（10）。
func TestLargeDenseTrainAccelBenchmark(t *testing.T) {
	skipUnlessAccelBench(t)
	rows := accelBenchEnvInt("LEAVES_BENCH_ROWS", 50000)
	cols := accelBenchEnvInt("LEAVES_BENCH_COLS", 64)
	rounds := accelBenchEnvInt("LEAVES_BENCH_ROUNDS", 10)
	dm := synthBenchDense(rows, cols)

	cases := []accelBenchCase{
		{name: "cpu_hist", accelMode: train.AccelModeCPU, treeMethod: train.TreeMethodHist, numThreads: 4},
		{name: "auto_hist", accelMode: train.AccelModeAuto, treeMethod: train.TreeMethodHist, numThreads: 4},
		{name: "auto_smart", accelMode: train.AccelModeAuto, treeMethod: train.TreeMethodAuto, numThreads: 4},
	}
	if tree.BornWebGPUAvailable() {
		cases = append(cases,
			accelBenchCase{
				name: "webgpu_hist", accelMode: train.AccelModeWebGPU,
				treeMethod: train.TreeMethodGPUHist, numThreads: 4,
			},
			accelBenchCase{
				name: "webgpu_t1", accelMode: train.AccelModeWebGPU,
				treeMethod: train.TreeMethodGPUHist, numThreads: 1,
			},
		)
	}
	if filtered := filterAccelBenchCases(cases); filtered == nil {
		t.Fatalf("unknown LEAVES_BENCH_ONLY=%q", os.Getenv("LEAVES_BENCH_ONLY"))
	} else {
		cases = filtered
	}

	t.Logf("large dense accel: rows=%d cols=%d rounds=%d webgpu=%v",
		rows, cols, rounds, tree.BornWebGPUAvailable())
	t.Log("mode\tseconds\trmse\thist_gpu%\tgain_gpu%\taccel_summary")

	for _, c := range cases {
		cfg := train.Config{
			Objective:     "reg:squarederror",
			NumRound:      rounds,
			MaxDepth:      6,
			LearningRate:  0.1,
			Lambda:        1,
			MaxBin:        64,
			TreeMethod:    c.treeMethod,
			AccelMode:     c.accelMode,
			NumThreads:    c.numThreads,
			HistBinPolicy: "global",
		}
		runAccelBenchCase(t, c, cfg, dm, func(t *testing.T, learner *train.Learner, dm data.Matrix) float64 {
			preds := make([]float64, dm.NumRow())
			_ = learner.PredictMargins(dm, preds)
			return rmseOnMatrix(dm, preds)
		})
	}
}

func rmseOnMatrix(dm data.Matrix, preds []float64) float64 {
	labels := dm.Labels()
	n := len(labels)
	if n == 0 {
		return 0
	}
	var se float64
	for i := 0; i < n; i++ {
		d := preds[i] - labels[i]
		se += d * d
	}
	return se / float64(n)
}
