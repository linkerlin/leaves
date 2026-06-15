package train_test

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/dmitryikh/leaves/data"
	"github.com/dmitryikh/leaves/metrics"
	"github.com/dmitryikh/leaves/train"
)

type rankXGBBaseline struct {
	Seed             int       `json:"seed"`
	Objective        string    `json:"objective"`
	EvalMetric       string    `json:"eval_metric"`
	NDCGK            int       `json:"ndcg_k"`
	NumRound         int       `json:"num_round"`
	MaxDepth         int       `json:"max_depth"`
	LearningRate     float64   `json:"learning_rate"`
	Lambda           float64   `json:"lambda"`
	TreeMethod       string    `json:"tree_method"`
	TrainNDCG        []float64 `json:"train_ndcg"`
	TestNDCG         []float64 `json:"test_ndcg"`
	FinalTrainNDCG   float64   `json:"final_train_ndcg"`
	FinalTestNDCG    float64   `json:"final_test_ndcg"`
	InitialTrainNDCG float64   `json:"initial_train_ndcg"`
	InitialTestNDCG  float64   `json:"initial_test_ndcg"`
}

type rankTrendGate struct {
	FinalTrainTol     float64
	MilestoneTol      float64
	TestFinalTol      float64
	MinTrainGain      float64
	MinFinalTrainNDCG float64
	ReachThresh       float64
	ReachRoundGap     int
	MinTestNDCG       float64
}

func loadRankXGBBaseline(t *testing.T, path string) rankXGBBaseline {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read baseline: %v", err)
	}
	var bl rankXGBBaseline
	if err := json.Unmarshal(b, &bl); err != nil {
		t.Fatalf("parse baseline: %v", err)
	}
	return bl
}

func ndcgOnMatrix(t *testing.T, dm data.Matrix, preds []float64, k int) float64 {
	t.Helper()
	name := "ndcg"
	opt := metrics.Options{Groups: groupsOf(dm)}
	if k > 0 {
		name = "ndcg@" + itoa(k)
		opt.NDCGK = k
	}
	m, err := metrics.Resolve(name, opt)
	if err != nil {
		t.Fatal(err)
	}
	v, err := m.Evaluate(dm.Labels(), preds)
	if err != nil {
		t.Fatal(err)
	}
	return v
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var buf [12]byte
	i := len(buf)
	for n > 0 {
		i--
		buf[i] = byte('0' + n%10)
		n /= 10
	}
	return string(buf[i:])
}

func fitRankLearner(t *testing.T, bl rankXGBBaseline, trainDM data.Matrix) *train.Learner {
	t.Helper()
	cfg := train.Config{
		Objective:    bl.Objective,
		NumRound:     bl.NumRound,
		MaxDepth:     bl.MaxDepth,
		LearningRate: bl.LearningRate,
		Lambda:       bl.Lambda,
		TreeMethod:   train.TreeMethodHist,
		Seed:         int64(bl.Seed),
		NDCGK:        bl.NDCGK,
	}
	if bl.NDCGK > 0 {
		cfg.EvalMetric = "ndcg@" + itoa(bl.NDCGK)
	} else {
		cfg.EvalMetric = train.EvalNDCG
	}
	learner, err := train.NewLearner(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := learner.Fit(trainDM); err != nil {
		t.Fatal(err)
	}
	return learner
}

func assertRankNDCGTrend(
	t *testing.T,
	bl rankXGBBaseline,
	leavesTrainHist []float64,
	leavesTestNDCG float64,
	gate rankTrendGate,
) {
	t.Helper()
	if len(leavesTrainHist) != bl.NumRound {
		t.Fatalf("metric history len %d, want %d", len(leavesTrainHist), bl.NumRound)
	}
	if math.Abs(leavesTrainHist[len(leavesTrainHist)-1]-bl.FinalTrainNDCG) > gate.FinalTrainTol {
		t.Errorf("final train ndcg: leaves=%f xgb=%f (tol %f)",
			leavesTrainHist[len(leavesTrainHist)-1], bl.FinalTrainNDCG, gate.FinalTrainTol)
	}

	trainGainLeaves := leavesTrainHist[len(leavesTrainHist)-1] - leavesTrainHist[0]
	trainGainXGB := bl.FinalTrainNDCG - bl.InitialTrainNDCG
	if trainGainLeaves < gate.MinTrainGain {
		t.Errorf("leaves train ndcg gain too small: %f (want >= %f)", trainGainLeaves, gate.MinTrainGain)
	}
	if trainGainXGB < gate.MinTrainGain {
		t.Errorf("xgb train ndcg gain too small: %f", trainGainXGB)
	}
	if gate.MinFinalTrainNDCG > 0 {
		if leavesTrainHist[len(leavesTrainHist)-1] < gate.MinFinalTrainNDCG || bl.FinalTrainNDCG < gate.MinFinalTrainNDCG {
			t.Errorf("final train ndcg should be >= %f: leaves=%f xgb=%f",
				gate.MinFinalTrainNDCG, leavesTrainHist[len(leavesTrainHist)-1], bl.FinalTrainNDCG)
		}
	}

	milestones := []int{0, 4, 9, 14, 19, bl.NumRound - 1}
	for _, idx := range milestones {
		if idx >= len(leavesTrainHist) {
			continue
		}
		if math.Abs(leavesTrainHist[idx]-bl.TrainNDCG[idx]) > gate.MilestoneTol {
			t.Errorf("round %d train ndcg: leaves=%f xgb=%f (tol %f)",
				idx+1, leavesTrainHist[idx], bl.TrainNDCG[idx], gate.MilestoneTol)
		}
	}

	if gate.ReachThresh > 0 {
		leavesReach := roundToReach(leavesTrainHist, gate.ReachThresh)
		xgbReach := roundToReach(bl.TrainNDCG, gate.ReachThresh)
		if leavesReach < 0 || xgbReach < 0 {
			t.Errorf("expected both to reach ndcg>=%f: leaves_round=%d xgb_round=%d",
				gate.ReachThresh, leavesReach, xgbReach)
		} else if diff := absInt(leavesReach - xgbReach); diff > gate.ReachRoundGap {
			t.Errorf("reach ndcg>=%f round gap too large: leaves=%d xgb=%d (tol %d)",
				gate.ReachThresh, leavesReach+1, xgbReach+1, gate.ReachRoundGap)
		}
	}

	if math.Abs(leavesTestNDCG-bl.FinalTestNDCG) > gate.TestFinalTol {
		t.Errorf("final test ndcg: leaves=%f xgb=%f (tol %f)", leavesTestNDCG, bl.FinalTestNDCG, gate.TestFinalTol)
	}
	if gate.MinTestNDCG > 0 && leavesTestNDCG < gate.MinTestNDCG {
		t.Errorf("leaves test ndcg too low: %f (want >= %f)", leavesTestNDCG, gate.MinTestNDCG)
	}
}

func runRankTrendCase(
	t *testing.T,
	trainPath, testPath, baselinePath string,
	skipHint string,
	gate rankTrendGate,
) {
	t.Helper()
	if _, err := os.Stat(trainPath); err != nil {
		t.Skipf("missing %s (%s)", trainPath, skipHint)
	}
	bl := loadRankXGBBaseline(t, baselinePath)
	trainDM, err := data.LoadRankingTSV(trainPath, "\t")
	if err != nil {
		t.Fatal(err)
	}
	testDM, err := data.LoadRankingTSV(testPath, "\t")
	if err != nil {
		t.Fatal(err)
	}

	learner := fitRankLearner(t, bl, trainDM)
	leavesTrainHist := learner.MetricHistory()

	testPreds := make([]float64, testDM.NumRow())
	if err := learner.PredictMargins(testDM, testPreds); err != nil {
		t.Fatal(err)
	}
	leavesTestNDCG := ndcgOnMatrix(t, testDM, testPreds, bl.NDCGK)
	assertRankNDCGTrend(t, bl, leavesTrainHist, leavesTestNDCG, gate)

	t.Logf("[%s] train %.4f->%.4f test leaves=%.4f xgb=%.4f",
		bl.Objective,
		leavesTrainHist[0], leavesTrainHist[len(leavesTrainHist)-1],
		leavesTestNDCG, bl.FinalTestNDCG)
}

// TestRankNDCGTrendVsXGBoost smoke 合成数据 rank:ndcg。
func TestRankNDCGTrendVsXGBoost(t *testing.T) {
	base := filepath.Join("..", "testdata")
	runRankTrendCase(t,
		filepath.Join(base, "rank_smoke_train.tsv"),
		filepath.Join(base, "rank_smoke_test.tsv"),
		filepath.Join(base, "rank_smoke_xgb_baseline.json"),
		"run testdata/gen_rank_smoke.py",
		rankTrendGate{
			FinalTrainTol: 0.08, MilestoneTol: 0.15, TestFinalTol: 0.15,
			MinTrainGain: 0.02, MinFinalTrainNDCG: 0.99,
			ReachThresh: 0.99, ReachRoundGap: 20, MinTestNDCG: 0.85,
		},
	)
}

// TestRankPairwiseTrendVsXGBoost smoke 合成数据 rank:pairwise（eval 仍为 ndcg）。
func TestRankPairwiseTrendVsXGBoost(t *testing.T) {
	base := filepath.Join("..", "testdata")
	runRankTrendCase(t,
		filepath.Join(base, "rank_smoke_train.tsv"),
		filepath.Join(base, "rank_smoke_test.tsv"),
		filepath.Join(base, "rank_smoke_pairwise_xgb_baseline.json"),
		"run testdata/gen_rank_smoke.py",
		rankTrendGate{
			FinalTrainTol: 0.10, MilestoneTol: 0.18, TestFinalTol: 0.15,
			MinTrainGain: 0.02, MinFinalTrainNDCG: 0.99,
			ReachThresh: 0.99, ReachRoundGap: 22, MinTestNDCG: 0.85,
		},
	)
}

// TestRankMSLTRNDCGTrendVsXGBoost MSLR-WEB10K 子集 rank:ndcg（生产向）。
func TestRankMSLTRNDCGTrendVsXGBoost(t *testing.T) {
	if testing.Short() {
		t.Skip("MSLTR rank trend is slow; skip with -short")
	}
	base := filepath.Join("..", "testdata")
	runRankTrendCase(t,
		filepath.Join(base, "rank_msltr_train.tsv"),
		filepath.Join(base, "rank_msltr_test.tsv"),
		filepath.Join(base, "rank_msltr_ndcg_xgb_baseline.json"),
		"run testdata/gen_rank_msltr.py",
		rankTrendGate{
			FinalTrainTol: 0.13, MilestoneTol: 0.15, TestFinalTol: 0.12,
			MinTrainGain: 0.05, MinTestNDCG: 0.28,
		},
	)
}

// TestRankMSLTRPairwiseTrendVsXGBoost MSLR-WEB10K 子集 rank:pairwise。
func TestRankMSLTRPairwiseTrendVsXGBoost(t *testing.T) {
	if testing.Short() {
		t.Skip("MSLTR rank trend is slow; skip with -short")
	}
	base := filepath.Join("..", "testdata")
	runRankTrendCase(t,
		filepath.Join(base, "rank_msltr_train.tsv"),
		filepath.Join(base, "rank_msltr_test.tsv"),
		filepath.Join(base, "rank_msltr_pairwise_xgb_baseline.json"),
		"run testdata/gen_rank_msltr.py",
		rankTrendGate{
			// pairwise 优化 RankNet 损失，train ndcg@10 与 XGB 可偏离较大；以 test 为主
			FinalTrainTol: 0.30, MilestoneTol: 0.30, TestFinalTol: 0.15,
			MinTrainGain: 0.08, MinTestNDCG: 0.30,
		},
	)
}

// TestRankMovieLensListwiseDemo MovieLens 100K listwise 对标 rank:ndcg（1–5 星相关性）。
func TestRankMovieLensListwiseDemo(t *testing.T) {
	base := filepath.Join("..", "testdata")
	runRankTrendCase(t,
		filepath.Join(base, "rank_movielens_train.tsv"),
		filepath.Join(base, "rank_movielens_test.tsv"),
		filepath.Join(base, "rank_movielens_ndcg_xgb_baseline.json"),
		"run testdata/gen_rank_movielens.py",
		rankTrendGate{
			FinalTrainTol: 0.10, MilestoneTol: 0.12, TestFinalTol: 0.10,
			MinTrainGain: 0.02, MinTestNDCG: 0.55,
		},
	)
}

// TestRankMovieLensPairwiseDemo MovieLens 100K rank:pairwise 平行对比。
func TestRankMovieLensPairwiseDemo(t *testing.T) {
	base := filepath.Join("..", "testdata")
	runRankTrendCase(t,
		filepath.Join(base, "rank_movielens_train.tsv"),
		filepath.Join(base, "rank_movielens_test.tsv"),
		filepath.Join(base, "rank_movielens_pairwise_xgb_baseline.json"),
		"run testdata/gen_rank_movielens.py",
		rankTrendGate{
			FinalTrainTol: 0.12, MilestoneTol: 0.15, TestFinalTol: 0.10,
			MinTrainGain: 0.015, MinTestNDCG: 0.55,
		},
	)
}

// TestRankMovieLensListwiseObjective rank:listwise 在 MovieLens 上（无 XGB 同名目标，验 NDCG@10）。
func TestRankMovieLensListwiseObjective(t *testing.T) {
	base := filepath.Join("..", "testdata")
	trainPath := filepath.Join(base, "rank_movielens_train.tsv")
	testPath := filepath.Join(base, "rank_movielens_test.tsv")
	if _, err := os.Stat(trainPath); err != nil {
		t.Skipf("missing %s (run testdata/gen_rank_movielens.py)", trainPath)
	}
	trainDM, err := data.LoadRankingTSV(trainPath, "\t")
	if err != nil {
		t.Fatal(err)
	}
	testDM, err := data.LoadRankingTSV(testPath, "\t")
	if err != nil {
		t.Fatal(err)
	}

	learner, err := train.NewLearner(train.Config{
		Objective:    train.ObjectiveRankListwise,
		NumRound:     40,
		MaxDepth:     4,
		LearningRate: 0.1,
		Lambda:       1.0,
		TreeMethod:   train.TreeMethodHist,
		Seed:         42,
		NDCGK:        10,
		EvalMetric:   "ndcg@10",
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := learner.Fit(trainDM); err != nil {
		t.Fatal(err)
	}

	testPreds := make([]float64, testDM.NumRow())
	if err := learner.PredictMargins(testDM, testPreds); err != nil {
		t.Fatal(err)
	}
	zeroPreds := make([]float64, testDM.NumRow())
	before := ndcgOnMatrix(t, testDM, zeroPreds, 10)
	after := ndcgOnMatrix(t, testDM, testPreds, 10)
	if after < before+0.02 {
		t.Errorf("test ndcg@10 should improve: before=%f after=%f", before, after)
	}
	if after < 0.55 {
		t.Errorf("test ndcg@10 too low: %f", after)
	}
	t.Logf("rank:listwise movielens test ndcg@10=%.4f (zero baseline=%.4f)", after, before)
}

func roundToReach(hist []float64, thresh float64) int {
	for i, v := range hist {
		if v >= thresh {
			return i
		}
	}
	return -1
}

func absInt(x int) int {
	if x < 0 {
		return -x
	}
	return x
}
