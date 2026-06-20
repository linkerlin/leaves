package recsys

import (
	"fmt"
	"os"
	"path/filepath"
)

// Workspace recsys 工作区目录布局。
type Workspace struct {
	Root string
}

// NewWorkspace 创建并确保根目录存在。
func NewWorkspace(root string) (Workspace, error) {
	if root == "" {
		return Workspace{}, fmt.Errorf("recsys: empty workspace root")
	}
	w := Workspace{Root: root}
	if err := w.Ensure(); err != nil {
		return Workspace{}, err
	}
	return w, nil
}

func (w Workspace) dir(parts ...string) string {
	all := append([]string{w.Root}, parts...)
	return filepath.Join(all...)
}

func (w Workspace) RawDir() string     { return w.dir("raw") }
func (w Workspace) CleanDir() string   { return w.dir("clean") }
func (w Workspace) CatalogDir() string { return w.dir("catalog") }
func (w Workspace) RecallDir() string  { return w.dir("recall") }
func (w Workspace) RankDir() string    { return w.dir("rank") }
func (w Workspace) ModelsDir() string  { return w.dir("models") }
func (w Workspace) DealDir() string    { return w.dir("deal") }
func (w Workspace) MetaDir() string    { return w.dir("meta") }

func (w Workspace) SamplesTrain() string { return filepath.Join(w.CleanDir(), "samples_train.tsv") }
func (w Workspace) SamplesTest() string  { return filepath.Join(w.CleanDir(), "samples_test.tsv") }
func (w Workspace) ItemsCatalog() string { return filepath.Join(w.CatalogDir(), "items.tsv") }
func (w Workspace) UserQID() string      { return filepath.Join(w.MetaDir(), "user_qid.tsv") }
func (w Workspace) PrepReport() string   { return filepath.Join(w.MetaDir(), "prep_report.json") }

func (w Workspace) RecallTrain() string { return filepath.Join(w.RecallDir(), "recall_train.tsv") }
func (w Workspace) RecallTest() string  { return filepath.Join(w.RecallDir(), "recall_test.tsv") }

func (w Workspace) RankTrain() string           { return filepath.Join(w.RankDir(), "rank_train.tsv") }
func (w Workspace) RankTest() string            { return filepath.Join(w.RankDir(), "rank_test.tsv") }
func (w Workspace) RankTrainManifest() string   { return filepath.Join(w.RankDir(), "rank_train_manifest.jsonl") }
func (w Workspace) RankTestManifest() string    { return filepath.Join(w.RankDir(), "rank_test_manifest.jsonl") }
func (w Workspace) RankTestScored() string      { return filepath.Join(w.RankDir(), "rank_test_scored.jsonl") }

func (w Workspace) ModelPath() string   { return filepath.Join(w.ModelsDir(), "model_rank_ndcg.leaves.json") }
func (w Workspace) RankEval() string    { return filepath.Join(w.MetaDir(), "rank_eval.json") }
func (w Workspace) DealTest() string    { return filepath.Join(w.DealDir(), "deal_test.tsv") }
func (w Workspace) DealLog() string     { return filepath.Join(w.DealDir(), "deal_log.jsonl") }

// Ensure 创建全部子目录。
func (w Workspace) Ensure() error {
	for _, d := range []string{
		w.RawDir(), w.CleanDir(), w.CatalogDir(), w.RecallDir(),
		w.RankDir(), w.ModelsDir(), w.DealDir(), w.MetaDir(),
	} {
		if err := os.MkdirAll(d, 0o755); err != nil {
			return fmt.Errorf("recsys: mkdir %s: %w", d, err)
		}
	}
	return nil
}
