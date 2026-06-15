package leaves

import (
	"github.com/linkerlin/leaves/data"
	"github.com/linkerlin/leaves/io"
	"github.com/linkerlin/leaves/train"
)

// TrainConfig 根包训练配置别名。
type TrainConfig = train.Config

// TrainLearner 根包训练器别名。
type TrainLearner = train.Learner

// Learner 与 TrainLearner 同义（v3.1 短名）。
type Learner = train.Learner

// EarlyStopping 早停。
type EarlyStopping = train.EarlyStopping

// LearningRateScheduler 学习率调度。
type LearningRateScheduler = train.LearningRateScheduler

// TrainingCallback 训练回调。
type TrainingCallback = train.TrainingCallback

// CallbackContext 回调上下文。
type CallbackContext = train.CallbackContext

// 常用目标函数名。
const (
	TrainObjectiveSquaredError   = train.ObjectiveSquaredError
	TrainObjectiveBinaryLogistic = train.ObjectiveBinaryLogistic
	TrainObjectiveTweedie        = train.ObjectiveTweedie
	TrainObjectiveSurvivalCox    = train.ObjectiveSurvivalCox
	TrainObjectiveSurvivalAFT    = train.ObjectiveSurvivalAFT
	TrainObjectiveRankNDCG       = train.ObjectiveRankNDCG
)

// 树方法。
const (
	TrainTreeMethodHist  = train.TreeMethodHist
	TrainTreeMethodExact = train.TreeMethodExact
	TrainTreeMethodAuto  = train.TreeMethodAuto
)

// NewTrainLearner 创建训练器。
func NewTrainLearner(cfg TrainConfig) (*TrainLearner, error) {
	return train.NewLearner(cfg)
}

// NewLearner 同 NewTrainLearner。
func NewLearner(cfg TrainConfig) (*Learner, error) {
	return train.NewLearner(cfg)
}

// ResumeFit 从 checkpoint 续训。
func ResumeFit(path string, cfg TrainConfig, dm data.Matrix) (*Learner, error) {
	return train.ResumeFit(path, cfg, dm)
}

// LoadCheckpoint 加载 checkpoint（不自动 Fit）。
func LoadCheckpoint(path string, cfg TrainConfig) (*Learner, error) {
	return train.LoadCheckpoint(path, cfg)
}

// FitExternal 外存矩阵训练。
func FitExternal(cfg TrainConfig, em data.ExternalMemoryMatrix) (*Learner, error) {
	return train.FitExternal(cfg, em)
}

// NewEarlyStopping 创建早停。
func NewEarlyStopping(rounds int, maximize bool) *EarlyStopping {
	return train.NewEarlyStopping(rounds, maximize)
}

// TrainDense 在 Dense 数据上训练并返回 Learner。
func TrainDense(vals []float64, rows, cols int, labels []float64, cfg TrainConfig) (*TrainLearner, error) {
	dm, err := data.NewDense(vals, rows, cols, labels, nil)
	if err != nil {
		return nil, err
	}
	learner, err := train.NewLearner(cfg)
	if err != nil {
		return nil, err
	}
	if err := learner.Fit(dm); err != nil {
		return nil, err
	}
	return learner, nil
}

// FileLoadOptions 训练数据文件加载选项（含自动嗅探）。
type FileLoadOptions = data.FileLoadOptions

// LoadData 从文件加载训练矩阵（自动嗅探格式）。
func LoadData(path string, opts FileLoadOptions) (data.Matrix, error) {
	return train.LoadData(path, opts)
}

// LoadDataAuto 使用默认嗅探选项加载训练数据。
func LoadDataAuto(path string) (data.Matrix, error) {
	return train.LoadDataAuto(path)
}

// FromFileAuto 同 LoadDataAuto（data 包别名）。
func FromFileAuto(path string) (data.Matrix, error) {
	return data.FromFileAuto(path)
}

// DetectFileFormat 嗅探训练数据格式。
func DetectFileFormat(path string) data.FileFormat {
	return data.DetectFileFormat(path)
}

// InferObjectiveFromModel 从模型文件推断 objective。
func InferObjectiveFromModel(path string) (string, error) {
	return train.InferObjectiveFromModel(path)
}

// NewLearnerFromFile 从数据文件创建并训练 Learner。
func NewLearnerFromFile(dataPath string, cfg TrainConfig, dataOpts FileLoadOptions) (*Learner, error) {
	return train.NewLearnerFromFile(dataPath, cfg, dataOpts)
}

// NewLearnerFromModelAndData 从模型 JSON 推断 objective（cfg.Objective 为空时）并训练。
func NewLearnerFromModelAndData(modelPath, dataPath string, cfg TrainConfig, dataOpts FileLoadOptions) (*Learner, error) {
	return train.NewLearnerFromModelAndData(modelPath, dataPath, cfg, dataOpts)
}

// SaveTrainModel 保存训练模型为 leaves.json。
func SaveTrainModel(path string, learner *TrainLearner, objective string) error {
	if learner == nil {
		return io.ErrFormatNotImplemented("nil learner")
	}
	return learner.Save(path)
}
