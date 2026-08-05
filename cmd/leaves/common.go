// Package main: leaves CLI 共享类型与辅助。
package main

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"time"

	"github.com/linkerlin/leaves/v2/data"
	"github.com/linkerlin/leaves/v2/metrics"
)

// paramsRecord 记录当次 train 全部 CLI 旋钮，供 Agent 从 runs.jsonl 完整复现。
// 字段与 skills/leaves-autotrain/cli.md、演进方案 WP-02 对齐。
type paramsRecord struct {
	Rounds         int     `json:"rounds"`
	Depth          int     `json:"depth"`
	MaxLeaves      int     `json:"max_leaves,omitempty"`
	LR             float64 `json:"lr"`
	Lambda         float64 `json:"lambda"`
	MinChildWeight float64 `json:"min_child_weight"`
	Gamma          float64 `json:"gamma"`
	MaxBin         int     `json:"max_bin"`
	Subsample      float64 `json:"subsample"`
	Colsample      float64 `json:"colsample"`
	TreeMethod     string  `json:"tree_method"`
	Seed           int64   `json:"seed"`
	NumClass       int     `json:"num_class,omitempty"`
	NumTarget      int     `json:"num_target,omitempty"`
	NDCGK          int     `json:"ndcg_k,omitempty"`
	EarlyStop      int     `json:"early_stop,omitempty"`
	CVFolds        int     `json:"cv_folds,omitempty"`
	EvalMetric     string  `json:"eval_metric,omitempty"`
}

// metricsSchemaVersion 是 metrics/sniff 等 Agent 信号 JSON 的契约版本（WP-18）。
const metricsSchemaVersion = 1

// metricsDoc 是 Agent 闭环的唯一信号契约（见 skills/leaves-autotrain/cli.md）。
type metricsDoc struct {
	SchemaVersion int       `json:"schema_version"`
	Objective     string    `json:"objective"`
	Metric        string    `json:"metric"`
	Value         float64   `json:"value"`
	Maximize      bool      `json:"maximize"`
	NRows         int       `json:"n_rows"`
	NFeatures     int       `json:"n_features,omitempty"`
	CVFolds       int       `json:"cv_folds,omitempty"`
	CVMean        float64   `json:"cv_mean,omitempty"`
	CVStd         float64   `json:"cv_std,omitempty"`
	FoldMetrics   []float64 `json:"fold_metrics,omitempty"`
	TrainMetric   float64   `json:"train_metric,omitempty"`
	BestRound     int       `json:"best_round,omitempty"`
	StoppedRound  int       `json:"stopped_round,omitempty"`
	ModelRound    int       `json:"model_round,omitempty"`
	// TrainAccel：Fit 后实际生效的训练加速模式（POST-13）；与推理 BackendAuto 无关。
	TrainAccel string `json:"train_accel,omitempty"`
	// FinalModel / FinalRound：--out-final 侧车（POST-12）；早停时为截断前 final-round。
	FinalModel string        `json:"final_model,omitempty"`
	FinalRound int           `json:"final_round,omitempty"`
	Params     *paramsRecord `json:"params,omitempty"`
}

// newParamsRecord 从 train CLI 旋钮组装完备 params（Agent 账本复现用）。
func newParamsRecord(
	rounds, depth, maxLeaves int,
	lr, lambda, minChildWeight, gamma float64,
	maxBin int,
	subsample, colsample float64,
	treeMethod string,
	seed int64,
	numClass, numTarget, ndcgK, earlyStop, cv int,
	evalMetric string,
) *paramsRecord {
	p := &paramsRecord{
		Rounds:         rounds,
		Depth:          depth,
		MaxLeaves:      maxLeaves,
		LR:             lr,
		Lambda:         lambda,
		MinChildWeight: minChildWeight,
		Gamma:          gamma,
		MaxBin:         maxBin,
		Subsample:      subsample,
		Colsample:      colsample,
		TreeMethod:     treeMethod,
		Seed:           seed,
		NumClass:       numClass,
		NumTarget:      numTarget,
		EarlyStop:      earlyStop,
		CVFolds:        cv,
		EvalMetric:     evalMetric,
	}
	// ndcg_k 仅对排序任务有意义；调用方在非排序时可传 0 省略。
	if ndcgK > 0 {
		p.NDCGK = ndcgK
	}
	return p
}

func writeMetrics(path string, doc metricsDoc) error {
	if doc.SchemaVersion == 0 {
		doc.SchemaVersion = metricsSchemaVersion
	}
	b, err := json.MarshalIndent(doc, "", "  ")
	if err != nil {
		return err
	}
	b = append(b, '\n')
	if path == "" || path == "-" {
		_, err = os.Stdout.Write(b)
		return err
	}
	_ = os.MkdirAll(filepath.Dir(path), 0o755)
	return os.WriteFile(path, b, 0o644)
}

// runRecord 是运行账本 runs.jsonl 的单行（Agent 跨迭代优化的持久记忆）。
type runRecord struct {
	Tag       string        `json:"tag"`
	TS        string        `json:"ts"`
	Model     string        `json:"model,omitempty"`
	Objective string        `json:"objective,omitempty"` // WP-17：供 --from-run 复现
	Metric    string        `json:"metric"`
	Value     float64       `json:"value"`
	Maximize  bool          `json:"maximize"`
	CVMean    float64       `json:"cv_mean,omitempty"`
	CVStd     float64       `json:"cv_std,omitempty"`
	Params    *paramsRecord `json:"params,omitempty"`
}

// appendRun 把本次训练记录追加到 JSONL 账本（不存在则创建）。
func appendRun(runsPath, tag, modelPath string, doc metricsDoc) error {
	if runsPath == "" {
		return nil
	}
	rec := runRecord{
		Tag:       tag,
		TS:        time.Now().UTC().Format(time.RFC3339),
		Model:     modelPath,
		Objective: doc.Objective,
		Metric:    doc.Metric,
		Value:     doc.Value,
		Maximize:  doc.Maximize,
		CVMean:    doc.CVMean,
		CVStd:     doc.CVStd,
		Params:    doc.Params,
	}
	b, err := json.Marshal(rec)
	if err != nil {
		return err
	}
	b = append(b, '\n')
	f, err := os.OpenFile(runsPath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return err
	}
	defer f.Close()
	_, err = f.Write(b)
	return err
}

func groupsOf(dm data.Matrix) []int {
	g, err := data.GroupsFromRanking(dm)
	if err != nil || len(g) == 0 {
		return nil
	}
	return g
}

// denseVals 把任意 Matrix（Dense/CSR）展平为行优先数组，供 PredictDense 使用。
func denseVals(dm data.Matrix) ([]float64, error) {
	n, c := dm.NumRow(), dm.NumCol()
	vals := make([]float64, n*c)
	buf := make([]float64, c)
	for i := 0; i < n; i++ {
		if err := dm.Row(i, buf); err != nil {
			return nil, fmt.Errorf("row %d: %w", i, err)
		}
		copy(vals[i*c:(i+1)*c], buf)
	}
	return vals, nil
}

func defaultMetric(objective string) string {
	switch objective {
	case "binary:logistic":
		return "logloss"
	case "multi:softmax", "multi:softprob":
		return "mlogloss"
	case "rank:ndcg", "rank:pairwise", "rank:listwise":
		return "ndcg@10"
	default:
		return "rmse"
	}
}

func metricMaximize(name string, numClass int, groups []int) bool {
	m, err := metrics.Resolve(name, metrics.Options{NumClass: numClass, Groups: groups})
	if err != nil {
		return false
	}
	return m.HigherIsBetter()
}

// metricInputs 按 objective（或从 metric 推断）把 raw margin 变换为 metric 输入。
func metricInputs(metric, objective string, margins []float64, numClass int) []float64 {
	switch objective {
	case "binary:logistic":
		out := make([]float64, len(margins))
		for i, x := range margins {
			out[i] = sigmoid(x)
		}
		return out
	case "multi:softmax", "multi:softprob":
		return softmaxRows(margins, numClass)
	case "count:poisson", "reg:gamma", "reg:tweedie":
		out := make([]float64, len(margins))
		for i, x := range margins {
			out[i] = math.Exp(x)
		}
		return out
	}
	// 未给 objective：按 metric 推断。
	switch metrics.NormalizeName(metric) {
	case "logloss", "auc", "error":
		out := make([]float64, len(margins))
		for i, x := range margins {
			out[i] = sigmoid(x)
		}
		return out
	case "mlogloss", "merror":
		if numClass > 1 {
			return softmaxRows(margins, numClass)
		}
	}
	return margins
}

func sigmoid(x float64) float64 {
	if x >= 0 {
		z := math.Exp(-x)
		return 1 / (1 + z)
	}
	z := math.Exp(x)
	return z / (1 + z)
}

func softmaxRows(margins []float64, numClass int) []float64 {
	if numClass <= 1 || len(margins)%numClass != 0 {
		return margins
	}
	n := len(margins) / numClass
	out := make([]float64, len(margins))
	for i := 0; i < n; i++ {
		row := margins[i*numClass : (i+1)*numClass]
		maxV := row[0]
		for _, v := range row[1:] {
			if v > maxV {
				maxV = v
			}
		}
		sum := 0.0
		for j, v := range row {
			e := math.Exp(v - maxV)
			out[i*numClass+j] = e
			sum += e
		}
		for j := range row {
			out[i*numClass+j] /= sum
		}
	}
	return out
}

func argmax(row []float64) int {
	best := 0
	for i := 1; i < len(row); i++ {
		if row[i] > row[best] {
			best = i
		}
	}
	return best
}
