package recall

import (
	"fmt"
	"math"
	"sort"

	"github.com/linkerlin/leaves/recsys"
)

// Config 召回参数。
type Config struct {
	PerUser int // 默认 100
	// MaxKnown 召回中最多保留的「已交互」Item 数（供 LTR 正样本标签）。
	// 剩余槽位只填未交互 Item，便于发牌 recent 过滤后仍有候选。
	// 0 表示默认 min(40, PerUser/2)。
	MaxKnown int
}

// Run 为 split 内各 User 生成 PerUser 条召回。
// 策略：先放部分已交互正样本（标签），再按 Tag/热度补未交互候选。
func Run(
	split string,
	samples []recsys.Interaction,
	catalog []recsys.CatalogItem,
	featNames []string,
	userQIDs []recsys.UserQID,
	cfg Config,
) ([]recsys.RecallRow, error) {
	if cfg.PerUser <= 0 {
		cfg.PerUser = 100
	}
	if cfg.MaxKnown <= 0 {
		cfg.MaxKnown = cfg.PerUser / 2
		if cfg.MaxKnown > 40 {
			cfg.MaxKnown = 40
		}
		if cfg.MaxKnown < 1 {
			cfg.MaxKnown = 1
		}
	}
	userSet := usersForSplit(userQIDs, split)
	if len(userSet) == 0 {
		return nil, fmt.Errorf("recall: no users for split %q", split)
	}

	catByID := map[string]recsys.CatalogItem{}
	for _, it := range catalog {
		catByID[it.Item] = it
	}
	tagIndex := buildTagIndex(catalog)
	globalByPop := sortedByPop(catalog)
	userPrefTags := preferredTags(samples)
	knownByUser := knownItems(samples)

	var out []recsys.RecallRow
	for u := range userSet {
		rows, err := recallOneUser(u, userPrefTags[u], knownByUser[u], catByID, tagIndex, globalByPop, cfg.PerUser, cfg.MaxKnown)
		if err != nil {
			return nil, err
		}
		out = append(out, rows...)
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].User != out[j].User {
			return out[i].User < out[j].User
		}
		return out[i].RecallScore > out[j].RecallScore
	})
	_ = featNames
	return out, nil
}

func usersForSplit(qids []recsys.UserQID, split string) map[string]int {
	m := map[string]int{}
	for _, r := range qids {
		if r.Split == split {
			m[r.User] = r.QID
		}
	}
	return m
}

func preferredTags(samples []recsys.Interaction) map[string]map[string]float64 {
	scores := map[string]map[string]float64{}
	for _, r := range samples {
		if scores[r.User] == nil {
			scores[r.User] = map[string]float64{}
		}
		scores[r.User][r.Tag] += r.Score
	}
	return scores
}

type knownItem struct {
	item  string
	score float64
	tag   string
}

func knownItems(samples []recsys.Interaction) map[string][]knownItem {
	// 同 (user,item) 取最高分
	best := map[string]map[string]knownItem{}
	for _, r := range samples {
		if best[r.User] == nil {
			best[r.User] = map[string]knownItem{}
		}
		prev, ok := best[r.User][r.Item]
		if !ok || r.Score > prev.score {
			best[r.User][r.Item] = knownItem{item: r.Item, score: r.Score, tag: r.Tag}
		}
	}
	out := map[string][]knownItem{}
	for u, m := range best {
		list := make([]knownItem, 0, len(m))
		for _, v := range m {
			list = append(list, v)
		}
		sort.Slice(list, func(i, j int) bool { return list[i].score > list[j].score })
		out[u] = list
	}
	return out
}

func buildTagIndex(catalog []recsys.CatalogItem) map[string][]recsys.CatalogItem {
	idx := map[string][]recsys.CatalogItem{}
	for _, it := range catalog {
		idx[it.Tag] = append(idx[it.Tag], it)
	}
	for tag := range idx {
		sort.Slice(idx[tag], func(i, j int) bool {
			return idx[tag][i].Feats[0] > idx[tag][j].Feats[0]
		})
	}
	return idx
}

func sortedByPop(catalog []recsys.CatalogItem) []recsys.CatalogItem {
	out := append([]recsys.CatalogItem(nil), catalog...)
	sort.Slice(out, func(i, j int) bool { return out[i].Feats[0] > out[j].Feats[0] })
	return out
}

func recallOneUser(
	user string,
	tagScores map[string]float64,
	known []knownItem,
	catByID map[string]recsys.CatalogItem,
	tagIndex map[string][]recsys.CatalogItem,
	global []recsys.CatalogItem,
	need, maxKnown int,
) ([]recsys.RecallRow, error) {
	type tagRank struct {
		tag   string
		score float64
	}
	var tags []tagRank
	for t, s := range tagScores {
		tags = append(tags, tagRank{t, s})
	}
	sort.Slice(tags, func(i, j int) bool { return tags[i].score > tags[j].score })
	if len(tags) == 0 {
		for t := range tagIndex {
			tags = append(tags, tagRank{t, 0})
		}
		sort.Slice(tags, func(i, j int) bool { return tags[i].tag < tags[j].tag })
	}

	seen := map[string]struct{}{}
	var out []recsys.RecallRow
	pick := func(it recsys.CatalogItem, rs float64) {
		if _, ok := seen[it.Item]; ok {
			return
		}
		seen[it.Item] = struct{}{}
		out = append(out, recsys.RecallRow{
			User: user, Item: it.Item, Tag: it.Tag,
			RecallScore: rs,
			Feats:       append([]float64(nil), it.Feats...),
		})
	}

	// 1) 已交互正样本（LTR 标签），限量
	nKnown := 0
	for _, k := range known {
		if nKnown >= maxKnown || len(out) >= need {
			break
		}
		it, ok := catByID[k.item]
		if !ok {
			// catalog 缺项时用交互 Tag + 零特征兜底
			it = recsys.CatalogItem{Item: k.item, Tag: k.tag, Feats: []float64{0, 0, 0, 0}}
		}
		// 召回分：偏好标签 + 星级
		rs := 1.0 + 0.2*k.score
		if len(it.Feats) > 0 {
			rs += 0.1 * it.Feats[0]
		}
		pick(it, rs)
		nKnown++
	}

	// 2) 未交互：按偏好 Tag 热度
	for len(out) < need {
		progress := false
		for _, tr := range tags {
			pool := tagIndex[tr.tag]
			for _, it := range pool {
				if len(out) >= need {
					break
				}
				if _, ok := seen[it.Item]; ok {
					continue
				}
				rs := 0.5*tr.score + 0.3*it.Feats[0] + 0.2*it.Feats[1]
				pick(it, rs)
				progress = true
				if len(out) >= need {
					break
				}
			}
		}
		if !progress {
			break
		}
	}
	// 3) 全局热门未交互补齐
	for _, it := range global {
		if len(out) >= need {
			break
		}
		if _, ok := seen[it.Item]; ok {
			continue
		}
		rs := 0.3*it.Feats[0] + 0.2*it.Feats[1]
		pick(it, rs)
	}
	if len(out) != need {
		return nil, fmt.Errorf("recall: user %s got %d items, want %d", user, len(out), need)
	}
	return out, nil
}

// Validate 校验每 User 恰 need 条且无 Item 重复。
func Validate(rows []recsys.RecallRow, need int) error {
	count := map[string]int{}
	items := map[string]map[string]struct{}{}
	for _, r := range rows {
		count[r.User]++
		if items[r.User] == nil {
			items[r.User] = map[string]struct{}{}
		}
		if _, ok := items[r.User][r.Item]; ok {
			return fmt.Errorf("recall: duplicate item %s for user %s", r.Item, r.User)
		}
		items[r.User][r.Item] = struct{}{}
	}
	for u, c := range count {
		if c != need {
			return fmt.Errorf("recall: user %s has %d items, want %d", u, c, need)
		}
	}
	return nil
}

// MaxRecallScore 辅助测试。
func MaxRecallScore(rows []recsys.RecallRow) float64 {
	max := -math.MaxFloat64
	for _, r := range rows {
		if r.RecallScore > max {
			max = r.RecallScore
		}
	}
	return max
}
