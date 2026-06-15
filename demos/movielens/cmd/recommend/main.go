// 加载已训练模型，对测试集用户生成 Top-K 电影推荐（按预测分排序）。
package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"

	_ "github.com/linkerlin/leaves" // 注册 leaves.json loader

	"github.com/linkerlin/leaves/data"
	"github.com/linkerlin/leaves/demos/movielens/rankutil"
	"github.com/linkerlin/leaves/io"
	"github.com/linkerlin/leaves/model"
)

const trainUserCount = 60

func main() {
	modelPath := flag.String("model", "", "leaves.json 模型路径")
	dataPath := flag.String("data", "", "测试 TSV（默认 testdata/rank_movielens_test.tsv）")
	groupIdx := flag.Int("group", 0, "测试集用户序号 [0,14]")
	qid := flag.Int("qid", -1, "显式 qid（覆盖 -group）")
	topK := flag.Int("topk", 10, "推荐条数")
	flag.Parse()

	dm, err := loadTestMatrix(*dataPath)
	if err != nil {
		fatal(err)
	}
	groups := dm.Groups()
	if *qid >= 0 {
		idx, err := groupIndexForQID(groups, *qid, trainUserCount)
		if err != nil {
			fatal(err)
		}
		*groupIdx = idx
	}
	if *groupIdx < 0 || *groupIdx >= len(groups) {
		fatal(fmt.Errorf("group %d 超出范围 [0,%d)", *groupIdx, len(groups)))
	}

	path := *modelPath
	if path == "" {
		path, err = defaultModelPath()
		if err != nil {
			fatal(err)
		}
	}
	m, err := io.LoadFromFile(path, &io.LoadOptions{LoadTransformation: false})
	if err != nil {
		fatal(fmt.Errorf("load model: %w", err))
	}
	defer m.Close()

	preds, err := predictGroup(m, dm, *groupIdx)
	if err != nil {
		fatal(err)
	}
	items, err := rankutil.RankGroup(dm, preds, *groupIdx, *topK)
	if err != nil {
		fatal(err)
	}

	userQID := rankutil.GroupQID(*groupIdx, trainUserCount)
	fmt.Printf("用户 qid=%d（测试集第 %d 位），Top-%d 推荐（label=历史星级 1–5）\n",
		userQID, *groupIdx, len(items))
	fmt.Println("rank  score      stars  row")
	for i, it := range items {
		fmt.Printf("%4d  %9.4f  %5.0f  %d\n", i+1, it.Score, it.Label, it.RowInGroup)
	}
}

func loadTestMatrix(path string) (*data.DenseWithGroups, error) {
	if path == "" {
		dataDir, err := rankutil.DataDir()
		if err != nil {
			return nil, err
		}
		path = filepath.Join(dataDir, "rank_movielens_test.tsv")
	}
	return data.LoadRankingTSV(path, "\t")
}

func defaultModelPath() (string, error) {
	outDir, err := rankutil.OutDir()
	if err != nil {
		return "", err
	}
	p := filepath.Join(outDir, "model_rank_ndcg.leaves.json")
	if _, err := os.Stat(p); err != nil {
		return "", fmt.Errorf("未指定 -model 且默认 %s 不存在（请先运行 train）", p)
	}
	return p, nil
}

func groupIndexForQID(groups []int, qid, trainUsers int) (int, error) {
	want := qid - trainUsers
	if want < 0 || want >= len(groups) {
		return 0, fmt.Errorf("qid %d 不在测试集范围 [%d,%d)",
			qid, trainUsers, trainUsers+len(groups)-1)
	}
	return want, nil
}

func predictGroup(m *model.Ensemble, dm *data.DenseWithGroups, groupIdx int) ([]float64, error) {
	start, count, err := rankutil.GroupSlice(dm, groupIdx)
	if err != nil {
		return nil, err
	}
	cols := dm.Cols
	if m.NFeatures() > cols {
		return nil, fmt.Errorf("模型特征数 %d > 数据列数 %d", m.NFeatures(), cols)
	}
	vals := dm.Data[start*cols : (start+count)*cols]
	out := make([]float64, count)
	if err := m.PredictDense(vals, count, cols, out, 0, 0); err != nil {
		return nil, err
	}
	// 填充完整 preds 切片供 RankGroup 使用
	full := make([]float64, dm.Rows)
	copy(full[start:start+count], out)
	return full, nil
}

func fatal(err error) {
	fmt.Fprintf(os.Stderr, "错误: %v\n", err)
	os.Exit(1)
}
