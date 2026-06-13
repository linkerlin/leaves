package train

import "github.com/dmitryikh/leaves/treebuilder"

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

// 目标函数名。
const (
	ObjectiveSquaredError    = "reg:squarederror"
	ObjectiveBinaryLogistic  = "binary:logistic"
	ObjectiveMultiSoftmax    = "multi:softmax"
	ObjectiveMultiSoftprob   = "multi:softprob"
	ObjectiveGamma           = "reg:gamma"
	ObjectivePoisson         = "count:poisson"
	ObjectiveRankPairwise    = "rank:pairwise"
	ObjectiveRankNDCG        = "rank:ndcg"
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
