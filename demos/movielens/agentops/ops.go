// Package agentops 为 MovieLens ranker demo 提供 Agent/MCP 共用操作与 JSON 契约。
package agentops

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	_ "github.com/linkerlin/leaves"

	"github.com/linkerlin/leaves/data"
	"github.com/linkerlin/leaves/demos/movielens/rankutil"
	"github.com/linkerlin/leaves/io"
	"github.com/linkerlin/leaves/recsys"
	"github.com/linkerlin/leaves/recsys/movielens"
	"github.com/linkerlin/leaves/recsys/pipeline"
	"github.com/linkerlin/leaves/recsys/tsvio"
	"github.com/linkerlin/leaves/train"
)

const (
	DefaultRounds = 40
	DefaultDepth  = 4
	DefaultLR     = 0.1
	DefaultLambda = 1.0
	DefaultNDCGK  = 10
	DefaultSeed   = 42
	TrainUsers    = 60
)

// Result 统一 Agent 可读结果（写入 metrics.json 或 MCP tool result）。
type Result struct {
	OK      bool           `json:"ok"`
	Op      string         `json:"op"`
	Message string         `json:"message,omitempty"`
	Error   string         `json:"error,omitempty"`
	Hint    string         `json:"hint,omitempty"`
	Data    map[string]any `json:"data,omitempty"`
	TS      string         `json:"ts"`
}

func okResult(op, msg string, data map[string]any) Result {
	return Result{OK: true, Op: op, Message: msg, Data: data, TS: time.Now().UTC().Format(time.RFC3339)}
}

func errResult(op, err, hint string) Result {
	return Result{OK: false, Op: op, Error: err, Hint: hint, TS: time.Now().UTC().Format(time.RFC3339)}
}

// Status 检查数据与模型是否就绪。
func Status() Result {
	dataDir, err := rankutil.DataDir()
	if err != nil {
		return errResult("status", err.Error(), "从仓库根目录运行，或设置 LEAVES_TESTDATA")
	}
	trainP := filepath.Join(dataDir, "rank_movielens_train.tsv")
	testP := filepath.Join(dataDir, "rank_movielens_test.tsv")
	_, trainOK := os.Stat(trainP)
	_, testOK := os.Stat(testP)
	outDir, _ := rankutil.OutDir()
	modelP := filepath.Join(outDir, "model_rank_ndcg.leaves.json")
	_, modelOK := os.Stat(modelP)
	return okResult("status", "MovieLens ranker workspace status", map[string]any{
		"data_dir":      dataDir,
		"train_tsv":     trainP,
		"test_tsv":      testP,
		"train_ready":   trainOK == nil,
		"test_ready":    testOK == nil,
		"out_dir":       outDir,
		"model_ndcg":    modelP,
		"model_ready":   modelOK == nil,
		"baseline_ndcg": filepath.Join(dataDir, "rank_movielens_ndcg_xgb_baseline.json"),
	})
}

// Prepare 确保 MovieLens ranking TSV 存在；缺失时尝试调用 gen_rank_movielens.py。
func Prepare(force bool) Result {
	st := Status()
	if st.OK && !force {
		d := st.Data
		if d["train_ready"] == true && d["test_ready"] == true {
			return okResult("prepare", "ranking TSV already present", d)
		}
	}
	dataDir, err := rankutil.DataDir()
	repoRoot := ""
	if err == nil {
		repoRoot = filepath.Dir(dataDir)
	} else {
		// 尝试 cwd 向上找
		cwd, _ := os.Getwd()
		repoRoot = findRepoRoot(cwd)
		dataDir = filepath.Join(repoRoot, "testdata")
	}
	script := filepath.Join(dataDir, "gen_rank_movielens.py")
	if _, err := os.Stat(script); err != nil {
		return errResult("prepare", "gen_rank_movielens.py not found: "+err.Error(),
			"cd testdata && python gen_rank_movielens.py（需 numpy xgboost 与网络下载 ml-100k）")
	}
	args := []string{script}
	if force {
		args = append(args, "--force")
	}
	cmd := exec.Command("python", args...)
	cmd.Dir = dataDir
	cmd.Stdout = os.Stderr
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		// 尝试 python3
		cmd = exec.Command("python3", args...)
		cmd.Dir = dataDir
		cmd.Stdout = os.Stderr
		cmd.Stderr = os.Stderr
		if err2 := cmd.Run(); err2 != nil {
			return errResult("prepare", fmt.Sprintf("python gen failed: %v / %v", err, err2),
				"安装 numpy xgboost；或手动 python testdata/gen_rank_movielens.py")
		}
	}
	return Status()
}

func findRepoRoot(start string) string {
	dir := start
	for i := 0; i < 8; i++ {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir
		}
		p := filepath.Dir(dir)
		if p == dir {
			break
		}
		dir = p
	}
	return start
}

// TrainParams 训练参数。
type TrainParams struct {
	Objective string `json:"objective"`
	Rounds    int    `json:"rounds"`
	Depth     int    `json:"depth"`
	LR        float64 `json:"lr"`
	Lambda    float64 `json:"lambda"`
	NDCGK     int    `json:"ndcg_k"`
	Seed      int64  `json:"seed"`
	OutModel  string `json:"out_model"`
	Metrics   string `json:"metrics"`
}

func defaultTrain() TrainParams {
	return TrainParams{
		Objective: train.ObjectiveRankNDCG,
		Rounds:    DefaultRounds,
		Depth:     DefaultDepth,
		LR:        DefaultLR,
		Lambda:    DefaultLambda,
		NDCGK:     DefaultNDCGK,
		Seed:      DefaultSeed,
	}
}

// Train 训练 ranker 并写 metrics。
func Train(p TrainParams) Result {
	def := defaultTrain()
	if p.Objective == "" {
		p.Objective = def.Objective
	}
	if p.Rounds <= 0 {
		p.Rounds = def.Rounds
	}
	if p.Depth <= 0 {
		p.Depth = def.Depth
	}
	if p.LR <= 0 {
		p.LR = def.LR
	}
	if p.Lambda <= 0 {
		p.Lambda = def.Lambda
	}
	if p.NDCGK <= 0 {
		p.NDCGK = def.NDCGK
	}
	if p.Seed == 0 {
		p.Seed = def.Seed
	}

	dataDir, err := rankutil.DataDir()
	if err != nil {
		return errResult("train", err.Error(), "先 tools/call movielens_prepare")
	}
	trainPath := filepath.Join(dataDir, "rank_movielens_train.tsv")
	testPath := filepath.Join(dataDir, "rank_movielens_test.tsv")
	trainDM, err := data.LoadRankingTSV(trainPath, "\t")
	if err != nil {
		return errResult("train", "load train: "+err.Error(), "movielens_prepare")
	}
	testDM, err := data.LoadRankingTSV(testPath, "\t")
	if err != nil {
		return errResult("train", "load test: "+err.Error(), "movielens_prepare")
	}

	cfg := train.Config{
		Objective:    p.Objective,
		NumRound:     p.Rounds,
		MaxDepth:     p.Depth,
		LearningRate: p.LR,
		Lambda:       p.Lambda,
		TreeMethod:   train.TreeMethodHist,
		Seed:         p.Seed,
		NDCGK:        p.NDCGK,
		EvalMetric:   fmt.Sprintf("ndcg@%d", p.NDCGK),
	}
	learner, err := train.NewLearner(cfg)
	if err != nil {
		return errResult("train", err.Error(), "")
	}
	if err := learner.Fit(trainDM); err != nil {
		return errResult("train", "fit: "+err.Error(), "降 rounds/depth 重试")
	}

	trainPred, err := rankutil.PredictMargins(learner, trainDM)
	if err != nil {
		return errResult("train", err.Error(), "")
	}
	testPred, err := rankutil.PredictMargins(learner, testDM)
	if err != nil {
		return errResult("train", err.Error(), "")
	}
	trainNDCG, _ := rankutil.NDCGAtK(trainDM, trainPred, p.NDCGK)
	testNDCG, _ := rankutil.NDCGAtK(testDM, testPred, p.NDCGK)

	outDir, err := rankutil.OutDir()
	if err != nil {
		return errResult("train", err.Error(), "")
	}
	outModel := p.OutModel
	if outModel == "" {
		outModel = filepath.Join(outDir, "model_"+sanitizeObj(p.Objective)+".leaves.json")
	}
	if err := io.SaveTrainModel(outModel, learner.Model(), p.Objective); err != nil {
		return errResult("train", "save: "+err.Error(), "")
	}

	metricsPath := p.Metrics
	if metricsPath == "" {
		metricsPath = filepath.Join(outDir, "metrics_train.json")
	}
	doc := map[string]any{
		"schema_version": 1,
		"op":             "train",
		"objective":      p.Objective,
		"metric":         fmt.Sprintf("ndcg@%d", p.NDCGK),
		"value":          testNDCG,
		"maximize":       true,
		"train_metric":   trainNDCG,
		"n_rows":         trainDM.NumRow(),
		"n_features":     trainDM.NumCol(),
		"n_groups":       len(trainDM.Groups()),
		"model":          outModel,
		"params": map[string]any{
			"rounds": p.Rounds, "depth": p.Depth, "lr": p.LR,
			"lambda": p.Lambda, "ndcg_k": p.NDCGK, "seed": p.Seed,
			"tree_method": "hist", "objective": p.Objective,
		},
	}
	if bl, err := rankutil.LoadXGBBaseline(rankutil.BaselinePath(dataDir, p.Objective)); err == nil {
		doc["xgb_baseline"] = map[string]any{
			"train_ndcg": bl.FinalTrainNDCG,
			"test_ndcg":  bl.FinalTestNDCG,
			"delta_test": testNDCG - bl.FinalTestNDCG,
		}
	}
	_ = writeJSON(metricsPath, doc)

	// runs.jsonl 追加
	runsPath := filepath.Join(outDir, "runs.jsonl")
	_ = appendJSONL(runsPath, map[string]any{
		"tag": p.Objective, "ts": time.Now().UTC().Format(time.RFC3339),
		"model": outModel, "metric": fmt.Sprintf("ndcg@%d", p.NDCGK),
		"value": testNDCG, "maximize": true, "params": doc["params"],
	})

	return okResult("train", fmt.Sprintf("trained %s test NDCG@%d=%.4f", p.Objective, p.NDCGK, testNDCG), map[string]any{
		"model":        outModel,
		"metrics":      metricsPath,
		"runs":         runsPath,
		"train_ndcg":   trainNDCG,
		"test_ndcg":    testNDCG,
		"n_features":   trainDM.NumCol(),
		"train_groups": len(trainDM.Groups()),
		"test_groups":  len(testDM.Groups()),
	})
}

// Eval 在测试集评估 NDCG。
func Eval(modelPath string, ndcgK int) Result {
	if ndcgK <= 0 {
		ndcgK = DefaultNDCGK
	}
	if modelPath == "" {
		outDir, err := rankutil.OutDir()
		if err != nil {
			return errResult("eval", err.Error(), "")
		}
		modelPath = filepath.Join(outDir, "model_rank_ndcg.leaves.json")
	}
	dataDir, err := rankutil.DataDir()
	if err != nil {
		return errResult("eval", err.Error(), "prepare data first")
	}
	testDM, err := data.LoadRankingTSV(filepath.Join(dataDir, "rank_movielens_test.tsv"), "\t")
	if err != nil {
		return errResult("eval", err.Error(), "")
	}
	// 用 train 包加载模型重评估：通过 learner 不便；用 io + rankutil
	// rankutil needs *train.Learner - use margins via model engine
	// Simpler: re-open via temporary train path - use io Load + predict per group

	// 复用 train 评估：构造临时 fit 不做，直接 load ensemble
	ens, err := io.LoadFromFile(modelPath, &io.LoadOptions{LoadTransformation: false})
	if err != nil {
		return errResult("eval", "load model: "+err.Error(), "先 movielens_train")
	}
	defer ens.Close()

	n := testDM.NumRow()
	nf := testDM.NumCol()
	vals := make([]float64, n*nf)
	buf := make([]float64, nf)
	for i := 0; i < n; i++ {
		_ = testDM.Row(i, buf)
		copy(vals[i*nf:(i+1)*nf], buf)
	}
	out := make([]float64, n*ens.NOutputGroups())
	if err := ens.PredictDense(vals, n, nf, out, 0, 0); err != nil {
		return errResult("eval", err.Error(), "")
	}
	preds := out
	if ens.NOutputGroups() > 1 {
		// 取第一输出组 margin
		preds = make([]float64, n)
		g := ens.NOutputGroups()
		for i := 0; i < n; i++ {
			preds[i] = out[i*g]
		}
	}
	ndcg, err := rankutil.NDCGAtK(testDM, preds, ndcgK)
	if err != nil {
		return errResult("eval", err.Error(), "")
	}

	outDir, _ := rankutil.OutDir()
	metricsPath := filepath.Join(outDir, "metrics_eval.json")
	doc := map[string]any{
		"schema_version": 1,
		"op":             "eval",
		"metric":         fmt.Sprintf("ndcg@%d", ndcgK),
		"value":          ndcg,
		"maximize":       true,
		"model":          modelPath,
		"n_rows":         n,
		"n_groups":       len(testDM.Groups()),
	}
	_ = writeJSON(metricsPath, doc)
	return okResult("eval", fmt.Sprintf("test NDCG@%d=%.4f", ndcgK, ndcg), map[string]any{
		"test_ndcg": ndcg, "metrics": metricsPath, "model": modelPath,
	})
}

// RecommendParams 推荐参数。
type RecommendParams struct {
	Model   string `json:"model"`
	Group   int    `json:"group"`
	QID     int    `json:"qid"`
	TopK    int    `json:"topk"`
	OutJSON string `json:"out_json"`
}

// Recommend 对测试用户输出 Top-K。
func Recommend(p RecommendParams) Result {
	if p.TopK <= 0 {
		p.TopK = 10
	}
	if p.QID == 0 {
		p.QID = -1
	}
	dataDir, err := rankutil.DataDir()
	if err != nil {
		return errResult("recommend", err.Error(), "")
	}
	testPath := filepath.Join(dataDir, "rank_movielens_test.tsv")
	dm, err := data.LoadRankingTSV(testPath, "\t")
	if err != nil {
		return errResult("recommend", err.Error(), "")
	}
	groups := dm.Groups()
	groupIdx := p.Group
	if p.QID >= 0 {
		want := p.QID - TrainUsers
		if want < 0 || want >= len(groups) {
			return errResult("recommend", fmt.Sprintf("qid %d out of test range", p.QID), "test qid start at 60")
		}
		groupIdx = want
	}
	if groupIdx < 0 || groupIdx >= len(groups) {
		return errResult("recommend", fmt.Sprintf("group %d out of range [0,%d)", groupIdx, len(groups)), "")
	}
	modelPath := p.Model
	if modelPath == "" {
		outDir, _ := rankutil.OutDir()
		modelPath = filepath.Join(outDir, "model_rank_ndcg.leaves.json")
	}
	ens, err := io.LoadFromFile(modelPath, &io.LoadOptions{LoadTransformation: false})
	if err != nil {
		return errResult("recommend", err.Error(), "movielens_train first")
	}
	defer ens.Close()

	// 预测该组
	start := 0
	for i := 0; i < groupIdx; i++ {
		start += groups[i]
	}
	gsize := groups[groupIdx]
	nf := dm.NumCol()
	vals := make([]float64, gsize*nf)
	buf := make([]float64, nf)
	for i := 0; i < gsize; i++ {
		_ = dm.Row(start+i, buf)
		copy(vals[i*nf:(i+1)*nf], buf)
	}
	out := make([]float64, gsize*ens.NOutputGroups())
	if err := ens.PredictDense(vals, gsize, nf, out, 0, 0); err != nil {
		return errResult("recommend", err.Error(), "")
	}
	preds := make([]float64, gsize)
	gOut := ens.NOutputGroups()
	if gOut < 1 {
		gOut = 1
	}
	for i := 0; i < gsize; i++ {
		preds[i] = out[i*gOut]
	}
	// 构造全长 pred 数组供 RankGroup
	allPred := make([]float64, dm.NumRow())
	copy(allPred[start:start+gsize], preds)
	items, err := rankutil.RankGroup(dm, allPred, groupIdx, p.TopK)
	if err != nil {
		return errResult("recommend", err.Error(), "")
	}
	userQID := rankutil.GroupQID(groupIdx, TrainUsers)
	metaRows, _ := rankutil.LoadTestMeta(dataDir)
	metaIdx := rankutil.MetaIndex(metaRows)
	list := make([]map[string]any, 0, len(items))
	for i, it := range items {
		rec := map[string]any{
			"rank": i + 1, "score": it.Score, "label": it.Label, "row": it.RowInGroup,
		}
		if m, ok := metaIdx[[2]int{userQID, it.RowInGroup}]; ok {
			rec["movie_id"] = m.MovieID
			rec["title"] = m.Title
			rec["user_id"] = m.UserID
		}
		list = append(list, rec)
	}
	note := "label=历史星级 1–5；row=组内行号"
	if len(metaRows) > 0 {
		note += "；含 movie_id/title（旁车 meta）"
	} else {
		note += "；无 meta 旁车时仅有 row（运行 gen_rank_movielens.py 生成 rank_movielens_*_meta.jsonl）"
	}
	payload := map[string]any{
		"qid": userQID, "group": groupIdx, "topk": len(list),
		"model": modelPath, "items": list,
		"note": note,
	}
	outDir, _ := rankutil.OutDir()
	outJSON := p.OutJSON
	if outJSON == "" {
		outJSON = filepath.Join(outDir, fmt.Sprintf("recommend_g%d.json", groupIdx))
	}
	_ = writeJSON(outJSON, payload)
	return okResult("recommend", fmt.Sprintf("Top-%d for qid=%d", len(list), userQID), map[string]any{
		"out_json": outJSON, "qid": userQID, "items": list,
	})
}

// FourStageParams MovieLens 四段流水线（prep→recall→rank→deal）参数。
type FourStageParams struct {
	Workspace  string
	TrainUsers int
	TestUsers  int
	RecallSize int
	Rounds     int
	DeckSize   int
	MaxSameTag int
	Seed       int64
	// SampleUser 若非空，在 data 中附带该用户发牌样本（含 title）
	SampleUser string
}

// FourStage 跑 recsys 四段：MovieLens → 准备 → 召回(100) → LTR → 发牌。
// 与 ranker-only full-pipeline 不同：走真实召回候选 + Tag 控重发牌。
func FourStage(p FourStageParams) Result {
	outDir, err := rankutil.OutDir()
	if err != nil {
		return errResult("four_stage", err.Error(), "从仓库根运行")
	}
	ws := p.Workspace
	if ws == "" {
		ws = filepath.Join(outDir, "fourstage")
	}
	if p.TrainUsers <= 0 {
		p.TrainUsers = 40
	}
	if p.TestUsers <= 0 {
		p.TestUsers = 10
	}
	if p.RecallSize <= 0 {
		p.RecallSize = 100
	}
	if p.Rounds <= 0 {
		p.Rounds = 20
	}
	if p.DeckSize <= 0 {
		p.DeckSize = 10
	}
	if p.MaxSameTag <= 0 {
		p.MaxSameTag = 3
	}
	if p.Seed == 0 {
		p.Seed = DefaultSeed
	}

	// outDir = <repo>/demos/movielens/out
	repoRoot := filepath.Dir(filepath.Dir(filepath.Dir(outDir)))

	mlCfg := movielens.DefaultConfig()
	mlCfg.RepoRoot = repoRoot
	mlCfg.Seed = p.Seed
	mlCfg.TrainUsers = p.TrainUsers
	mlCfg.TestUsers = p.TestUsers

	ds, titles, err := movielens.Load(mlCfg)
	if err != nil {
		return errResult("four_stage", err.Error(),
			"首次需下载 ml-100k：网络可达或预置 .cache/ml-100k.zip；也可 go run ./recsys/cmd/movielens")
	}

	w, err := recsys.NewWorkspace(ws)
	if err != nil {
		return errResult("four_stage", err.Error(), "")
	}
	if err := movielens.WriteTitles(filepath.Join(w.MetaDir(), "item_titles.tsv"), titles); err != nil {
		return errResult("four_stage", err.Error(), "")
	}

	cfg := recsys.DefaultSmokeConfig()
	cfg.Seed = p.Seed
	cfg.TrainUsers = p.TrainUsers
	cfg.TestUsers = p.TestUsers
	cfg.RecallSize = p.RecallSize
	cfg.TrainRounds = p.Rounds
	cfg.DeckSize = p.DeckSize
	cfg.MaxSameTag = p.MaxSameTag
	cfg.NumItems = len(ds.Catalog)

	res, err := pipeline.RunFromDataset(w, ds, cfg)
	if err != nil {
		return errResult("four_stage", err.Error(), "检查 recsys 四段日志；catalog 是否覆盖 samples")
	}

	// 解析发牌样本（第一用户或指定用户）+ 片名
	sample, sampleUser := sampleDeal(w.DealTest(), titles, p.SampleUser)
	reportPath := filepath.Join(w.MetaDir(), "four_stage_report.json")
	dataOut := map[string]any{
		"workspace":     w.Root,
		"train_users":   res.Prep.TrainUsers,
		"test_users":    res.Prep.TestUsers,
		"catalog_size":  res.Prep.CatalogSize,
		"recall_train":  res.RecallTrain,
		"recall_test":   res.RecallTest,
		"rank_train":    res.RankTrain,
		"rank_test":     res.RankTest,
		"ndcg_k":        res.Eval.NDCGK,
		"train_ndcg":    res.Eval.TrainNDCG,
		"test_ndcg":     res.Eval.TestNDCG,
		"deal_rows":     res.DealRows,
		"deal_tsv":      w.DealTest(),
		"model":         w.ModelPath(),
		"item_titles":   filepath.Join(w.MetaDir(), "item_titles.tsv"),
		"sample_user":   sampleUser,
		"sample_deal":   sample,
		"report":        reportPath,
	}
	_ = writeJSON(reportPath, dataOut)
	return okResult("four_stage",
		fmt.Sprintf("MovieLens four-stage done: test NDCG@%d=%.4f, deal=%d rows",
			res.Eval.NDCGK, res.Eval.TestNDCG, res.DealRows),
		dataOut)
}

func sampleDeal(dealPath string, titles map[string]string, wantUser string) ([]map[string]any, string) {
	rows, err := tsvio.ReadDeal(dealPath)
	if err != nil || len(rows) == 0 {
		return nil, ""
	}
	user := wantUser
	if user == "" {
		user = rows[0].User
	}
	var out []map[string]any
	for _, r := range rows {
		if r.User != user {
			continue
		}
		rec := map[string]any{
			"rank": r.Rank, "item": r.Item, "tag": r.Tag, "score": r.Score,
		}
		if t, ok := titles[r.Item]; ok {
			rec["title"] = t
		}
		out = append(out, rec)
	}
	return out, user
}

// FullPipeline prepare → train → eval → recommend。
func FullPipeline(p TrainParams, rec RecommendParams) Result {
	pr := Prepare(false)
	if !pr.OK {
		return pr
	}
	tr := Train(p)
	if !tr.OK {
		return tr
	}
	modelPath, _ := tr.Data["model"].(string)
	ev := Eval(modelPath, p.NDCGK)
	if !ev.OK {
		return ev
	}
	if rec.Model == "" {
		rec.Model = modelPath
	}
	rc := Recommend(rec)
	if !rc.OK {
		return rc
	}
	outDir, _ := rankutil.OutDir()
	reportPath := filepath.Join(outDir, "pipeline_report.json")
	report := map[string]any{
		"ok": true, "op": "full_pipeline",
		"prepare": pr.Data, "train": tr.Data, "eval": ev.Data, "recommend": rc.Data,
		"ts": time.Now().UTC().Format(time.RFC3339),
	}
	_ = writeJSON(reportPath, report)
	return okResult("full_pipeline", "MovieLens ranker pipeline complete", map[string]any{
		"report": reportPath, "train": tr.Data, "eval": ev.Data, "recommend": rc.Data,
	})
}

func sanitizeObj(obj string) string {
	return strings.ReplaceAll(obj, ":", "_")
}

func writeJSON(path string, v any) error {
	b, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return err
	}
	b = append(b, '\n')
	_ = os.MkdirAll(filepath.Dir(path), 0o755)
	return os.WriteFile(path, b, 0o644)
}

func appendJSONL(path string, v any) error {
	b, err := json.Marshal(v)
	if err != nil {
		return err
	}
	_ = os.MkdirAll(filepath.Dir(path), 0o755)
	f, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return err
	}
	defer f.Close()
	_, err = f.Write(append(b, '\n'))
	return err
}

// WriteResult 打印 JSON 结果（Agent 读 stdout）。
func WriteResult(r Result) {
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	_ = enc.Encode(r)
}
