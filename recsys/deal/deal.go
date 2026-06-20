package deal

import (
	"encoding/json"
	"fmt"
	"os"
	"sort"

	"github.com/linkerlin/leaves/recsys"
)

// Config 发牌策略。
type Config struct {
	DeckSize   int
	MaxSameTag int
}

// DefaultConfig 默认发牌参数。
func DefaultConfig() Config {
	return Config{DeckSize: 10, MaxSameTag: 3}
}

// LogEntry 发牌日志行。
type LogEntry struct {
	User            string `json:"User"`
	InputCandidates int    `json:"input_candidates"`
	AfterDedup      int    `json:"after_dedup"`
	AfterTagFilter  int    `json:"after_tag_filter"`
	DroppedRecent   int    `json:"dropped_recent"`
	DroppedTag      int    `json:"dropped_tag"`
	Overflow        bool   `json:"overflow"`
}

// Run 对 scored manifest 按 User 发牌。
func Run(
	scored []recsys.ManifestRow,
	recentByUser map[string]map[string]struct{},
	cfg Config,
) ([]recsys.DealRow, []LogEntry, error) {
	if cfg.DeckSize <= 0 {
		cfg.DeckSize = 10
	}
	if cfg.MaxSameTag <= 0 {
		cfg.MaxSameTag = 3
	}
	byUser := groupScored(scored)
	users := sortedUsers(byUser)

	var all []recsys.DealRow
	var logs []LogEntry
	for _, user := range users {
		cands := byUser[user]
		sort.Slice(cands, func(i, j int) bool { return cands[i].Score > cands[j].Score })
		recent := recentByUser[user]
		if recent == nil {
			recent = map[string]struct{}{}
		}

		var filtered []recsys.ManifestRow
		droppedRecent := 0
		for _, c := range cands {
			if _, seen := recent[c.Item]; seen {
				droppedRecent++
				continue
			}
			filtered = append(filtered, c)
		}

		deck, droppedTag, overflow := pickWithTagLimit(filtered, cfg)
		for i := range deck {
			all = append(all, recsys.DealRow{
				User: user, Item: deck[i].Item, Tag: deck[i].Tag,
				Score: deck[i].Score, Rank: i + 1,
			})
		}
		logs = append(logs, LogEntry{
			User:            user,
			InputCandidates: len(cands),
			AfterDedup:      len(filtered),
			AfterTagFilter:  len(deck),
			DroppedRecent:   droppedRecent,
			DroppedTag:      droppedTag,
			Overflow:        overflow,
		})
	}
	return all, logs, nil
}

func groupScored(rows []recsys.ManifestRow) map[string][]recsys.ManifestRow {
	m := map[string][]recsys.ManifestRow{}
	for _, r := range rows {
		m[r.User] = append(m[r.User], r)
	}
	return m
}

func sortedUsers(m map[string][]recsys.ManifestRow) []string {
	out := make([]string, 0, len(m))
	for u := range m {
	 out = append(out, u)
	}
	sort.Strings(out)
	return out
}

func pickWithTagLimit(cands []recsys.ManifestRow, cfg Config) (deck []recsys.ManifestRow, droppedTag int, overflow bool) {
	tagCnt := map[string]int{}
	for _, c := range cands {
		if tagCnt[c.Tag] >= cfg.MaxSameTag {
			droppedTag++
			continue
		}
		deck = append(deck, c)
		tagCnt[c.Tag]++
		if len(deck) >= cfg.DeckSize {
			return deck, droppedTag, false
		}
	}
	if len(deck) < cfg.DeckSize {
		overflow = fillOverflow(&deck, cands, cfg.DeckSize)
	}
	return deck, droppedTag, overflow
}

func fillOverflow(deck *[]recsys.ManifestRow, cands []recsys.ManifestRow, need int) bool {
	have := map[string]struct{}{}
	for _, d := range *deck {
		have[d.Item] = struct{}{}
	}
	for _, c := range cands {
		if len(*deck) >= need {
			break
		}
		if _, ok := have[c.Item]; ok {
			continue
		}
		*deck = append(*deck, c)
		have[c.Item] = struct{}{}
		return true
	}
	return false
}

// RecentItems 从 samples 构建用户已览 Item 集合。
func RecentItems(samples []recsys.Interaction) map[string]map[string]struct{} {
	out := map[string]map[string]struct{}{}
	for _, r := range samples {
		if out[r.User] == nil {
			out[r.User] = map[string]struct{}{}
		}
		out[r.User][r.Item] = struct{}{}
	}
	return out
}

// WriteLog 写发牌日志 JSONL。
func WriteLog(path string, logs []LogEntry) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	enc := json.NewEncoder(f)
	for _, l := range logs {
		if err := enc.Encode(l); err != nil {
			return fmt.Errorf("deal: write log: %w", err)
		}
	}
	return nil
}

// Validate 校验发牌结果。
func Validate(rows []recsys.DealRow, recent map[string]map[string]struct{}, maxSameTag, deckSize int) error {
	byUser := map[string][]recsys.DealRow{}
	for _, r := range rows {
		byUser[r.User] = append(byUser[r.User], r)
	}
	for user, deck := range byUser {
		if len(deck) > deckSize {
			return fmt.Errorf("deal: user %s deck size %d > %d", user, len(deck), deckSize)
		}
		tagCnt := map[string]int{}
		seenItem := map[string]struct{}{}
		for _, d := range deck {
			if _, ok := seenItem[d.Item]; ok {
				return fmt.Errorf("deal: duplicate item %s for %s", d.Item, user)
			}
			seenItem[d.Item] = struct{}{}
			tagCnt[d.Tag]++
			if rec, ok := recent[user]; ok {
				if _, hit := rec[d.Item]; hit {
					return fmt.Errorf("deal: recent item %s exposed for %s", d.Item, user)
				}
			}
		}
		for tag, c := range tagCnt {
			if c > maxSameTag+1 { // +1 for overflow fill
				return fmt.Errorf("deal: user %s tag %s count %d", user, tag, c)
			}
		}
	}
	return nil
}
