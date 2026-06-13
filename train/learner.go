package train

import (
	"fmt"
	"math"

	"github.com/dmitryikh/leaves/booster"
	"github.com/dmitryikh/leaves/data"
	"github.com/dmitryikh/leaves/metrics"
	"github.com/dmitryikh/leaves/model"
	"github.com/dmitryikh/leaves/objective"
	"github.com/dmitryikh/leaves/tree"
	"github.com/dmitryikh/leaves/treebuilder"
)

// Config 训练配置（T1/T2/T3）。
type Config struct {
	Booster         string
	Objective       string
	NumClass        int
	NumRound        int
	MaxDepth        int
	LearningRate    float64
	Lambda          float64
	MinHessian      float64
	Gamma           float64
	MaxBin          int
	TreeMethod      string
	EvalMetric      string
	Subsample       float64
	ColsampleByTree float64
	Seed            int64
	NumThreads      int // 0 = 全部 CPU；T4 多线程 hist
	NumParallelTree int
	AccelMode           string // auto|webgpu|born_cpu|cpu；空则 LEAVES_TRAIN_ACCEL
	HistBinPolicy       string // global|per_node；hist 路径默认 global
	MonotoneConstraints []int  // 每特征 -1/0/1，对标 XGBoost monotone_constraints
	EvalSet             data.Matrix
	EarlyStop           *EarlyStopping
	CheckpointEvery     int
	CheckpointPath      string
	DART                *booster.DARTConfig
	// 排序学习（T5，对标 XGBoost LambdaMART）
	NDCGK          int  // eval / lambda ndcg@k；0=全量
	LambdaRankNorm bool // lambdarank_norm，默认 true
	MaxPosition    int  // max_position；0=不截断
}

// Learner 训练编排器。
type Learner struct {
	cfg                Config
	obj                objective.Func
	booster            booster.Booster
	numGroups          int
	metric             metrics.Metric
	metricHistory      []float64
	resolvedTreeMethod  string
	useGPUHist          bool
	effectiveAccelMode  string
	accelLogged         bool
	marginEngine       *tree.BornEngine
	marginGPULogged    bool
	marginPredictGPU   int
	marginPredictCPU   int
}

// NewLearner 创建 Learner。
func NewLearner(cfg Config) (*Learner, error) {
	obj, err := objective.ByNameWithClass(cfg.Objective, cfg.NumClass)
	if err != nil {
		return nil, err
	}
	rankCfg := objective.RankTrainConfig{
		NDCGK:       cfg.NDCGK,
		LambdaNorm:  lambdaRankNormDefault(cfg),
		MaxPosition: cfg.MaxPosition,
	}
	if _, ok := objective.IsRanking(obj); ok {
		obj = objective.ConfigureRanking(obj, rankCfg)
		cfg.Subsample = 1.0 // 排序训练需 query 完整（对标 XGBoost group）
	}
	if cfg.NumRound <= 0 {
		cfg.NumRound = 10
	}
	if cfg.LearningRate <= 0 {
		cfg.LearningRate = 0.3
	}
	if cfg.Booster == "" {
		cfg.Booster = BoosterGBTree
	}
	if cfg.TreeMethod == "" {
		cfg.TreeMethod = treebuilder.MethodAuto
	}
	if cfg.Subsample <= 0 {
		cfg.Subsample = 1.0
	}
	if cfg.ColsampleByTree <= 0 {
		cfg.ColsampleByTree = 1.0
	}
	numGroups := 1
	if mc, ok := objective.IsMulticlass(obj); ok {
		numGroups = mc.Classes()
	}
	metric, err := evalMetricFor(cfg, numGroups)
	if err != nil {
		return nil, err
	}
	if cfg.EarlyStop != nil && metric != nil {
		cfg.EarlyStop.Maximize = metricMaximize(metric)
	}
	return &Learner{cfg: cfg, obj: obj, numGroups: numGroups, metric: metric}, nil
}

// Fit 在 Matrix 上训练。
func (l *Learner) Fit(dm data.Matrix) error {
	if dm == nil {
		return fmt.Errorf("train: nil matrix")
	}
	l.beginTrainAccel(dm)
	defer func() {
		l.endTrainAccel()
		l.closeMarginEngine()
	}()
	if rankObj, ok := objective.IsRanking(l.obj); ok {
		if _, err := data.GroupsFromRanking(dm); err != nil {
			return fmt.Errorf("train: ranking requires GroupedMatrix: %w", err)
		}
		return l.fitRanking(dm, rankObj)
	}
	labels := dm.Labels()
	n := dm.NumRow()
	g := l.numGroups

	if err := l.initBooster(dm, labels); err != nil {
		return err
	}
	l.metricHistory = nil

	predSize := n * g
	preds := make([]float64, predSize)
	grad := make([]float64, predSize)
	hess := make([]float64, predSize)
	evalPreds := make([]float64, predSize)
	predRow := make([]float64, g)
	gradRow := make([]float64, g)
	hessRow := make([]float64, g)

	mc, isMC := objective.IsMulticlass(l.obj)

	for round := 0; round < l.cfg.NumRound; round++ {
		l.predictMarginsInternal(dm, preds, false)
		if isMC {
			for i := 0; i < n; i++ {
				copy(predRow, preds[i*g:(i+1)*g])
				w := data.WeightAt(dm, i)
				mc.GradHessVec(predRow, labels[i], w, gradRow, hessRow)
				copy(grad[i*g:(i+1)*g], gradRow)
				copy(hess[i*g:(i+1)*g], hessRow)
			}
		} else {
			for i := 0; i < n; i++ {
				w := data.WeightAt(dm, i)
				p := preds[i*g]
				grad[i], hess[i] = l.obj.GradHess(p, labels[i], w)
			}
		}
		l.booster.Boost(dm, grad, hess)
		if l.metric != nil {
			l.predictMarginsInternal(dm, evalPreds, false)
			metricLabels, metricPreds := metricInputs(l.cfg, labels, evalPreds, g)
			if v, err := evaluateTrainMetric(l, metricLabels, metricPreds, dm); err == nil {
				l.metricHistory = append(l.metricHistory, v)
			}
		}
		if l.cfg.EvalSet != nil && l.cfg.EarlyStop != nil {
			if score, err := evalMetricOnSet(l, l.cfg.EvalSet); err == nil {
				if l.cfg.EarlyStop.update(score, round+1) {
					break
				}
			}
		}
		if l.cfg.CheckpointEvery > 0 && l.cfg.CheckpointPath != "" && (round+1)%l.cfg.CheckpointEvery == 0 {
			_ = SaveCheckpointFile(l.cfg.CheckpointPath, round+1, l)
		}
	}
	return nil
}

func (l *Learner) initBooster(dm data.Matrix, labels []float64) error {
	tbCfg := l.treebuilderCfg(dm)
	trainParams := booster.TrainParams{
		Subsample:       l.cfg.Subsample,
		ColsampleByTree: l.cfg.ColsampleByTree,
		Seed:            l.cfg.Seed,
		NumParallelTree: l.cfg.NumParallelTree,
		DART:            l.cfg.DART,
	}
	method := l.resolvedTreeMethod
	if method == "" {
		method = l.cfg.TreeMethod
	}

	switch l.cfg.Booster {
	case BoosterGBLinear:
		base := l.obj.InitialPred(labels, dm.Weights())
		l.booster = booster.NewGBLinear(dm.NumCol(), l.numGroups, base, booster.GBLinearConfig{
			LearningRate: l.cfg.LearningRate,
			Lambda:       l.cfg.Lambda,
		})
		return nil
	default:
		base := l.obj.InitialPred(labels, dm.Weights())
		if mc, ok := objective.IsMulticlass(l.obj); ok {
			vec := mc.InitialPredVec(labels, dm.Weights())
			if len(vec) > 0 {
				base = vec[0]
			}
			l.booster = booster.NewGBTree(dm.NumCol(), base, l.numGroups, tbCfg, method, trainParams)
			if gt, ok := l.booster.(*booster.GBTree); ok {
				gt.Forest().BaseScores = vec
			}
			return nil
		}
		l.booster = booster.NewGBTree(dm.NumCol(), base, 1, tbCfg, method, trainParams)
		return nil
	}
}

func metricInputs(cfg Config, labels, margins []float64, numGroups int) ([]float64, []float64) {
	switch cfg.Objective {
	case ObjectiveBinaryLogistic:
		probs := make([]float64, len(labels))
		for i, m := range margins {
			probs[i] = sigmoid(m)
		}
		return labels, probs
	case ObjectiveMultiSoftmax, ObjectiveMultiSoftprob:
		n := len(labels)
		probs := make([]float64, n*numGroups)
		for i := 0; i < n; i++ {
			row := margins[i*numGroups : (i+1)*numGroups]
			p := softmaxRow(row)
			copy(probs[i*numGroups:(i+1)*numGroups], p)
		}
		return labels, probs
	case ObjectivePoisson, ObjectiveGamma:
		vals := make([]float64, len(margins))
		for i, m := range margins {
			vals[i] = math.Exp(m)
		}
		return labels, vals
	default:
		return labels, margins
	}
}

func softmaxRow(row []float64) []float64 {
	out := make([]float64, len(row))
	maxV := row[0]
	for _, v := range row[1:] {
		if v > maxV {
			maxV = v
		}
	}
	sum := 0.0
	for i, v := range row {
		e := math.Exp(v - maxV)
		out[i] = e
		sum += e
	}
	if sum > 0 {
		inv := 1 / sum
		for i := range out {
			out[i] *= inv
		}
	}
	return out
}

// MetricHistory 返回每轮 eval metric。
func (l *Learner) MetricHistory() []float64 {
	if l.metricHistory == nil {
		return nil
	}
	out := make([]float64, len(l.metricHistory))
	copy(out, l.metricHistory)
	return out
}

// Model 返回训练完成的 ModelIR。
func (l *Learner) Model() *model.ModelIR {
	if l.booster == nil {
		return nil
	}
	switch b := l.booster.(type) {
	case *booster.GBLinear:
		lin := b.Linear()
		return &model.ModelIR{
			Kind:             model.KindGBLinear,
			NumFeatures:      lin.NumFeatures,
			NRawOutputGroups: lin.NumOutputGroups,
			NOutputGroups:    lin.NumOutputGroups,
			Name:             lin.Name,
			Linear:           lin,
		}
	case *booster.GBTree:
		f := b.Forest()
		kind := model.KindGBTree
		name := f.Name
		if name == "leaves.dart" {
			kind = model.KindDART
		}
		return &model.ModelIR{
			Kind:             kind,
			NumFeatures:      f.NumFeatures,
			NRawOutputGroups: l.numGroups,
			NOutputGroups:    l.numGroups,
			Name:             name,
			Forest:           f,
		}
	default:
		return nil
	}
}

// PredictMargins 用当前模型预测 raw margin。
func (l *Learner) PredictMargins(dm data.Matrix, out []float64) error {
	if l.booster == nil {
		return fmt.Errorf("train: not fitted")
	}
	need := dm.NumRow() * l.numGroups
	if len(out) < need {
		return fmt.Errorf("train: output too short (need %d)", need)
	}
	l.predictMarginsInternal(dm, out, true)
	return nil
}

func sigmoid(x float64) float64 {
	if x >= 0 {
		z := math.Exp(-x)
		return 1 / (1 + z)
	}
	z := math.Exp(x)
	return z / (1 + z)
}
