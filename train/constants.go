package train

import "github.com/linkerlin/leaves/treebuilder"

// Booster 类型。
const (
	BoosterGBTree   = "gbtree"
	BoosterGBLinear = "gblinear"
)

// 树构建算法。
const (
	TreeMethodHist     = treebuilder.MethodHist
	TreeMethodExact    = treebuilder.MethodExact
	TreeMethodAuto     = treebuilder.MethodAuto
	TreeMethodGPUHist  = treebuilder.MethodGPUHist
)

// 训练加速模式（LEAVES_TRAIN_ACCEL / Config.AccelMode）。
const (
	AccelModeAuto    = treebuilder.AccelModeAuto
	AccelModeWebGPU  = treebuilder.AccelModeWebGPU
	AccelModeBornCPU = treebuilder.AccelModeBornCPU
	AccelModeCPU     = treebuilder.AccelModeCPU
)

// 目标函数名。
const (
	ObjectiveSquaredError    = "reg:squarederror"
	ObjectiveBinaryLogistic  = "binary:logistic"
	ObjectiveMultiSoftmax    = "multi:softmax"
	ObjectiveMultiSoftprob   = "multi:softprob"
	ObjectiveGamma           = "reg:gamma"
	ObjectivePoisson         = "count:poisson"
	ObjectiveTweedie         = "reg:tweedie"
	ObjectiveSurvivalCox     = "survival:cox"
	ObjectiveSurvivalAFT     = "survival:aft"
	ObjectiveRankPairwise    = "rank:pairwise"
	ObjectiveRankNDCG        = "rank:ndcg"
	ObjectiveRankListwise     = "rank:listwise"
)

// LambdaRank 配对策略（对标 XGBoost lambdarank_pair_method）。
const (
	LambdaRankPairFull = "full"
	LambdaRankPairTopK = "topk"
	LambdaRankPairMean = "mean"
)

// 评估指标名（与 metrics.Resolve / XGBoost eval_metric 对齐）。
const (
	EvalRMSE     = "rmse"
	EvalMAE      = "mae"
	EvalMAPE     = "mape"
	EvalRMSLE    = "rmsle"
	EvalAUC      = "auc"
	EvalLogLoss  = "logloss"
	EvalError    = "error"
	EvalMLogLoss = "mlogloss"
	EvalMError   = "merror"
	EvalNDCG     = "ndcg"
	EvalMAP      = "map"
)
