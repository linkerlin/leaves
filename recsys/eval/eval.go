// Package eval 提供召回/排序/发牌三层离线评估与业务可配置阈值门禁
// （演进方案 §17.5 / RC-04）。阈值缺失时结论只能是 exploratory，不得进入 candidate。
package eval

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"sort"

	"github.com/linkerlin/leaves/v2/recsys"
	"github.com/linkerlin/leaves/v2/recsys/contract"
)

const SchemaVersion = 1

// 层名（与 ReleaseEvidence.Gates 对齐）。
const (
	LayerData          = "data"
	LayerCandidateRank = "candidate_rank"
	LayerDeal          = "deal"
)

// Threshold 单指标阈值；Min/Max 至少设一个，Level 决定越界时的门禁级别。
type Threshold struct {
	Layer string   `json:"layer"`
	Name  string   `json:"name"`
	Min   *float64 `json:"min,omitempty"`
	Max   *float64 `json:"max,omitempty"`
	Level string   `json:"level"` // warn | block
}

// Config 评估配置。Thresholds 为空 → Report.Purpose=exploratory、不做门禁。
type Config struct {
	RecallK    int         `json:"recall_k"`
	NDCGK      int         `json:"ndcg_k"`
	DeckSize   int         `json:"deck_size"`
	MaxSameTag int         `json:"max_same_tag"`
	Thresholds []Threshold `json:"thresholds,omitempty"`
}

// DefaultConfig 默认评估参数（不含阈值——业务必须显式给出）。
func DefaultConfig() Config {
	return Config{RecallK: 10, NDCGK: 10, DeckSize: 10, MaxSameTag: 3}
}

// RankGroup 一个 subject 的排序组：按模型排序后的标签序列。
type RankGroup struct {
	SubjectID string
	Labels    []float64 // 命中位次对齐；0 表示未命中
}

// Inputs 评估输入（全部来自流水线工件）。
type Inputs struct {
	// Relevant: 评估期内每个 subject 的真实正样本集合（ground truth）。
	Relevant map[string]map[string]bool
	// Ranked: 每个 subject 的候选列表（按分数降序）。
	Ranked map[string][]string
	// CatalogSize 目录物品数（coverage 分母）。
	CatalogSize int
	// Groups: 排序标签组（NDCG/MAP 输入）。
	Groups []RankGroup
	// Deals + cold-start 分层（可选）。
	Deals     []recsys.DealRow
	ColdUsers map[string]bool
	// 数据层：as-of 因果检查结果（split.CheckLeakage 的计数）。
	DataLeakCount  int
	DataEventCount int
}

// Report evaluation.json 内容。
type Report struct {
	SchemaVersion int                           `json:"schema_version"`
	Purpose       string                        `json:"purpose"` // exploratory | gate
	Metrics       map[string]float64            `json:"metrics"`
	Strata        map[string]map[string]float64 `json:"strata,omitempty"`
	Gates         []contract.GateResult         `json:"gates,omitempty"`
	Status        string                        `json:"status"` // ok | warn | block | exploratory
}

// RecallAtK = 命中正样本 / 全部正样本，对每个 subject 平均。
func RecallAtK(relevant map[string]map[string]bool, ranked map[string][]string, k int) float64 {
	if k <= 0 || len(relevant) == 0 {
		return 0
	}
	sum := 0.0
	n := 0
	for subj, rel := range relevant {
		n++
		if len(rel) == 0 {
			continue
		}
		hit := 0
		list := ranked[subj]
		limit := k
		if len(list) < limit {
			limit = len(list)
		}
		for i := 0; i < limit; i++ {
			if rel[list[i]] {
				hit++
			}
		}
		sum += float64(hit) / float64(len(rel))
	}
	return sum / float64(n)
}

// NDCGAtK 二值增益 NDCG@K，组平均。
func NDCGAtK(groups []RankGroup, k int) float64 {
	if k <= 0 || len(groups) == 0 {
		return 0
	}
	sum := 0.0
	for _, g := range groups {
		dcg := 0.0
		idcg := 0.0
		hits := 0
		limit := k
		if len(g.Labels) < limit {
			limit = len(g.Labels)
		}
		for i := 0; i < limit; i++ {
			gain := 0.0
			if g.Labels[i] > 0 {
				gain = 1
				hits++
			}
			dcg += gain / math.Log2(float64(i)+2)
			if i < hits { // 理想排列：正样本全排前
				idcg += 1 / math.Log2(float64(i)+2)
			}
		}
		if idcg > 0 {
			sum += dcg / idcg
		}
	}
	return sum / float64(len(groups))
}

// MAPAtK 二值 AP@K，组平均。
func MAPAtK(groups []RankGroup, k int) float64 {
	if k <= 0 || len(groups) == 0 {
		return 0
	}
	sum := 0.0
	for _, g := range groups {
		hit := 0
		precSum := 0.0
		total := 0
		for _, l := range g.Labels {
			if l > 0 {
				total++
			}
		}
		if total == 0 {
			continue
		}
		limit := k
		if len(g.Labels) < limit {
			limit = len(g.Labels)
		}
		for i := 0; i < limit; i++ {
			if g.Labels[i] > 0 {
				hit++
				precSum += float64(hit) / float64(i+1)
			}
		}
		sum += precSum / float64(total)
	}
	return sum / float64(len(groups))
}

// Coverage = ranked 去重物品数 / 目录大小。
func Coverage(ranked map[string][]string, catalogSize int) float64 {
	if catalogSize <= 0 {
		return 0
	}
	seen := map[string]bool{}
	for _, list := range ranked {
		for _, it := range list {
			seen[it] = true
		}
	}
	return float64(len(seen)) / float64(catalogSize)
}

// Compute 汇总三层指标。
func Compute(cfg Config, in Inputs) *Report {
	r := &Report{SchemaVersion: SchemaVersion, Metrics: map[string]float64{}}
	m := r.Metrics

	m["recall_at_k"] = RecallAtK(in.Relevant, in.Ranked, cfg.RecallK)
	m["coverage"] = Coverage(in.Ranked, in.CatalogSize)
	m["ndcg_at_k"] = NDCGAtK(in.Groups, cfg.NDCGK)
	m["map_at_k"] = MAPAtK(in.Groups, cfg.NDCGK)
	if in.DataEventCount > 0 {
		m["data_leakage_rate"] = float64(in.DataLeakCount) / float64(in.DataEventCount)
	} else {
		m["data_leakage_rate"] = float64(in.DataLeakCount)
	}

	// 分层（cold vs returning）。
	if len(in.ColdUsers) > 0 {
		r.Strata = map[string]map[string]float64{}
		for name, want := range map[string]bool{"cold": true, "returning": false} {
			rel, rk := map[string]map[string]bool{}, map[string][]string{}
			for s, v := range in.Relevant {
				if in.ColdUsers[s] == want {
					rel[s] = v
				}
			}
			for s, v := range in.Ranked {
				if in.ColdUsers[s] == want {
					rk[s] = v
				}
			}
			r.Strata[name] = map[string]float64{"recall_at_k": RecallAtK(rel, rk, cfg.RecallK)}
		}
	}

	// 发牌质量层。
	if len(in.Deals) > 0 {
		fill, dup, overflow, itemCov := DeckQuality(in.Deals, cfg.DeckSize, cfg.MaxSameTag, in.CatalogSize)
		m["deck_fill_rate"] = fill
		m["deck_dup_rate"] = dup
		m["deck_tag_overflow_rate"] = overflow
		m["deck_item_coverage"] = itemCov
	}
	return r
}

// DeckQuality 发牌层指标：deck 满足率、重复率、Tag 溢出率、物品覆盖。
func DeckQuality(deals []recsys.DealRow, deckSize, maxSameTag, catalogSize int) (fill, dup, overflow, itemCov float64) {
	if len(deals) == 0 || deckSize <= 0 {
		return 0, 0, 0, 0
	}
	byUser := map[string][]recsys.DealRow{}
	items := map[string]bool{}
	dupRows := 0
	for _, d := range deals {
		byUser[d.User] = append(byUser[d.User], d)
		items[d.Item] = true
	}
	// 重复行检测
	counted := map[string]int{}
	for _, d := range deals {
		key := d.User + "\x00" + d.Item
		counted[key]++
		if counted[key] > 1 {
			dupRows++
		}
	}
	overflowUsers := 0
	fillSum := 0.0
	for _, deck := range byUser {
		fillSum += float64(len(deck)) / float64(deckSize)
		tagCnt := map[string]int{}
		bad := false
		for _, d := range deck {
			tagCnt[d.Tag]++
			if tagCnt[d.Tag] > maxSameTag {
				bad = true
			}
		}
		if bad {
			overflowUsers++
		}
	}
	users := float64(len(byUser))
	fill = fillSum / users
	dup = float64(dupRows) / float64(len(deals))
	overflow = float64(overflowUsers) / users
	if catalogSize > 0 {
		itemCov = float64(len(items)) / float64(catalogSize)
	}
	return
}

// Gate 按阈值评估指标。未知指标 → block（metric_not_computed）。
func Gate(cfg Config, r *Report) {
	if len(cfg.Thresholds) == 0 {
		r.Purpose = "exploratory"
		r.Status = "exploratory"
		return
	}
	r.Purpose = "gate"
	r.Status = contract.StatusOK
	for _, th := range cfg.Thresholds {
		v, ok := r.Metrics[th.Name]
		gr := contract.GateResult{Layer: th.Layer, Name: th.Name, Status: contract.StatusOK}
		switch {
		case !ok:
			gr.Status = contract.StatusBlock
			gr.Reason = "metric_not_computed"
		case th.Min != nil && v < *th.Min, th.Max != nil && v > *th.Max:
			gr.Status = th.Level
			gr.Reason = fmt.Sprintf("value %v outside [%v, %v]", v, deref(th.Min), deref(th.Max))
		}
		r.Gates = append(r.Gates, gr)
		if gr.Status == contract.StatusBlock {
			r.Status = contract.StatusBlock
		} else if gr.Status == contract.StatusWarn && r.Status != contract.StatusBlock {
			r.Status = contract.StatusWarn
		}
	}
}

func deref(p *float64) (v any) {
	if p == nil {
		return "-inf"
	}
	return *p
}

// Evaluate 计算并门禁。
func Evaluate(cfg Config, in Inputs) *Report {
	r := Compute(cfg, in)
	Gate(cfg, r)
	return r
}

// WriteReport 写 evaluation.json。
func WriteReport(path string, r *Report) error {
	b, err := json.MarshalIndent(r, "", "  ")
	if err != nil {
		return fmt.Errorf("eval: marshal: %w", err)
	}
	if err := os.WriteFile(path, b, 0o644); err != nil {
		return fmt.Errorf("eval: write %s: %w", path, err)
	}
	return nil
}

// RankViews 从流水线工件构造评估视图：
// relevant（正样本集）、ranked（按 margin 降序的候选列表）、Groups（标签序列）。
// 供 CLI 与端到端演练共用。
func RankViews(testSamples []recsys.Interaction, scored []recsys.ManifestRow) (
	relevant map[string]map[string]bool, ranked map[string][]string, groups []RankGroup,
) {
	relevant = map[string]map[string]bool{}
	for _, s := range testSamples {
		if relevant[s.User] == nil {
			relevant[s.User] = map[string]bool{}
		}
		relevant[s.User][s.Item] = true
	}
	byUser := map[string][]recsys.ManifestRow{}
	for _, r := range scored {
		byUser[r.User] = append(byUser[r.User], r)
	}
	ranked = map[string][]string{}
	for u, rows := range byUser {
		sorted := append([]recsys.ManifestRow(nil), rows...)
		sort.Slice(sorted, func(i, j int) bool { return sorted[i].Score > sorted[j].Score })
		list := make([]string, len(sorted))
		labels := make([]float64, len(sorted))
		for i, r := range sorted {
			list[i] = r.Item
			if relevant[u][r.Item] {
				labels[i] = 1
			}
		}
		ranked[u] = list
		groups = append(groups, RankGroup{SubjectID: u, Labels: labels})
	}
	return relevant, ranked, groups
}

// SortedMetricNames 稳定输出指标名（测试/报告用）。
func SortedMetricNames(m map[string]float64) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}
