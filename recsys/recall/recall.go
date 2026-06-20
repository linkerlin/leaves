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
}

// Run 为 split 内各 User 生成 PerUser 条召回。
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
	userSet := usersForSplit(userQIDs, split)
	if len(userSet) == 0 {
		return nil, fmt.Errorf("recall: no users for split %q", split)
	}

	tagIndex := buildTagIndex(catalog)
	globalByPop := sortedByPop(catalog)
	userPrefTags := preferredTags(samples)

	var out []recsys.RecallRow
	for u := range userSet {
		rows, err := recallOneUser(u, userPrefTags[u], tagIndex, globalByPop, cfg.PerUser)
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
	tagIndex map[string][]recsys.CatalogItem,
	global []recsys.CatalogItem,
	need int,
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

	// 轮询偏好 Tag，按 feat_pop 取 Item
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
	// 全局热门补齐
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
	var max float64 = -math.MaxFloat64
	for _, r := range rows {
		if r.RecallScore > max {
			max = r.RecallScore
		}
	}
	return max
}
