package train

import (
	"bytes"
	"fmt"

	"github.com/linkerlin/leaves/booster"
	"github.com/linkerlin/leaves/data"
	leavesio "github.com/linkerlin/leaves/io"
	"github.com/linkerlin/leaves/treebuilder"
)

// LoadCheckpoint 从 checkpoint JSON 恢复 Learner，并从已完成轮次续训。
func LoadCheckpoint(path string, cfg Config) (*Learner, error) {
	round, obj, modelJSON, err := LoadCheckpointFile(path)
	if err != nil {
		return nil, err
	}
	if cfg.Objective == "" && obj != "" {
		cfg.Objective = obj
	}
	ir, _, err := leavesio.ParseLeavesJSON(bytes.NewReader(modelJSON))
	if err != nil {
		return nil, fmt.Errorf("train: checkpoint model: %w", err)
	}
	if ir.Forest == nil {
		return nil, fmt.Errorf("train: checkpoint requires gbtree model")
	}
	learner, err := NewLearner(cfg)
	if err != nil {
		return nil, err
	}
	tbCfg := treebuilder.Config{
		MaxDepth:     cfg.MaxDepth,
		MinHessian:   cfg.MinHessian,
		Lambda:       cfg.Lambda,
		Gamma:        cfg.Gamma,
		LearningRate: cfg.LearningRate,
		MaxBin:       cfg.MaxBin,
		MaxLeaves:    cfg.MaxLeaves,
	}
	method := cfg.TreeMethod
	if method == "" {
		method = treebuilder.MethodAuto
	}
	trainParams := booster.TrainParams{
		Subsample:       cfg.Subsample,
		ColsampleByTree: cfg.ColsampleByTree,
		Seed:            cfg.Seed,
		NumParallelTree: cfg.NumParallelTree,
		DART:            cfg.DART,
	}
	learner.booster = booster.NewGBTreeFromForest(ir.Forest, tbCfg, method, trainParams)
	learner.resumeFromRound = round
	if cfg.NumRound <= round {
		cfg.NumRound = round + 1
		learner.cfg.NumRound = cfg.NumRound
	}
	return learner, nil
}

// ResumeFit 加载 checkpoint 并继续训练。
func ResumeFit(path string, cfg Config, dm data.Matrix) (*Learner, error) {
	learner, err := LoadCheckpoint(path, cfg)
	if err != nil {
		return nil, err
	}
	if err := learner.Fit(dm); err != nil {
		return nil, err
	}
	return learner, nil
}
