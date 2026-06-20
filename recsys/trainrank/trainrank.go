package trainrank

import (
	"fmt"

	"github.com/linkerlin/leaves/data"
	"github.com/linkerlin/leaves/demos/movielens/rankutil"
	"github.com/linkerlin/leaves/io"
	"github.com/linkerlin/leaves/recsys"
	"github.com/linkerlin/leaves/recsys/tsvio"
	"github.com/linkerlin/leaves/train"
)

// EvalResult 训练评估结果。
type EvalResult struct {
	TrainNDCG float64 `json:"train_ndcg"`
	TestNDCG  float64 `json:"test_ndcg"`
	NDCGK     int     `json:"ndcg_k"`
	Rounds    int     `json:"rounds"`
	Objective string  `json:"objective"`
}

// Run 训练 rank:ndcg 并写模型与 test scored manifest。
func Run(w recsys.Workspace, cfg recsys.SmokeConfig) (EvalResult, error) {
	trainDM, err := data.LoadRankingTSV(w.RankTrain(), "\t")
	if err != nil {
		return EvalResult{}, fmt.Errorf("trainrank: load train: %w", err)
	}
	testDM, err := data.LoadRankingTSV(w.RankTest(), "\t")
	if err != nil {
		return EvalResult{}, fmt.Errorf("trainrank: load test: %w", err)
	}

	learnerCfg := train.Config{
		Objective:    train.ObjectiveRankNDCG,
		NumRound:     cfg.TrainRounds,
		MaxDepth:     4,
		LearningRate: 0.1,
		Lambda:       1.0,
		TreeMethod:   train.TreeMethodHist,
		Seed:         cfg.Seed,
		NDCGK:        cfg.NDCGK,
		EvalMetric:   fmt.Sprintf("ndcg@%d", cfg.NDCGK),
	}
	learner, err := train.NewLearner(learnerCfg)
	if err != nil {
		return EvalResult{}, err
	}
	if err := learner.Fit(trainDM); err != nil {
		return EvalResult{}, err
	}

	trainPred, err := rankutil.PredictMargins(learner, trainDM)
	if err != nil {
		return EvalResult{}, err
	}
	testPred, err := rankutil.PredictMargins(learner, testDM)
	if err != nil {
		return EvalResult{}, err
	}
	trainNDCG, err := rankutil.NDCGAtK(trainDM, trainPred, cfg.NDCGK)
	if err != nil {
		return EvalResult{}, err
	}
	testNDCG, err := rankutil.NDCGAtK(testDM, testPred, cfg.NDCGK)
	if err != nil {
		return EvalResult{}, err
	}

	if err := io.SaveTrainModel(w.ModelPath(), learner.Model(), learnerCfg.Objective); err != nil {
		return EvalResult{}, err
	}

	manifest, err := tsvio.ReadManifestJSONL(w.RankTestManifest())
	if err != nil {
		return EvalResult{}, err
	}
	if len(manifest) != testDM.NumRow() {
		return EvalResult{}, fmt.Errorf("trainrank: manifest rows %d != test rows %d", len(manifest), testDM.NumRow())
	}
	for i := range manifest {
		manifest[i].Score = testPred[i]
	}
	if err := tsvio.WriteManifestJSONL(w.RankTestScored(), manifest); err != nil {
		return EvalResult{}, err
	}

	eval := EvalResult{
		TrainNDCG: trainNDCG,
		TestNDCG:  testNDCG,
		NDCGK:     cfg.NDCGK,
		Rounds:    cfg.TrainRounds,
		Objective: learnerCfg.Objective,
	}
	if err := tsvio.WriteJSON(w.RankEval(), eval); err != nil {
		return EvalResult{}, err
	}
	return eval, nil
}

// ScoreWithModel 加载已训模型对 test 集打分（供测试复用）。
func ScoreWithModel(w recsys.Workspace) error {
	m, err := io.LoadFromFile(w.ModelPath(), &io.LoadOptions{LoadTransformation: false})
	if err != nil {
		return err
	}
	defer m.Close()
	testDM, err := data.LoadRankingTSV(w.RankTest(), "\t")
	if err != nil {
		return err
	}
	out := make([]float64, testDM.NumRow())
	if err := m.PredictDense(testDM.Data, testDM.NumRow(), testDM.Cols, out, 0, 0); err != nil {
		return err
	}
	manifest, err := tsvio.ReadManifestJSONL(w.RankTestManifest())
	if err != nil {
		return err
	}
	for i := range manifest {
		manifest[i].Score = out[i]
	}
	return tsvio.WriteManifestJSONL(w.RankTestScored(), manifest)
}
