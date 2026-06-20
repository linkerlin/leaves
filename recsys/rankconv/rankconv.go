package rankconv

import (
	"bufio"
	"fmt"
	"os"
	"sort"
	"strconv"
	"strings"

	"github.com/linkerlin/leaves/recsys"
)

// Result 排序 TSV 转换产物。
type Result struct {
	Manifest []recsys.ManifestRow
	Rows     int
}

// Run 将召回 TSV 转为 leaves ranking TSV + manifest。
func Run(
	recallRows []recsys.RecallRow,
	samples []recsys.Interaction,
	userQIDs []recsys.UserQID,
	split string,
	rankPath, manifestPath string,
) (Result, error) {
	qidByUser := map[string]int{}
	for _, u := range userQIDs {
		if u.Split == split {
			qidByUser[u.User] = u.QID
		}
	}
	labelMap := interactionLabels(samples)

	byUser := map[string][]recsys.RecallRow{}
	for _, r := range recallRows {
		byUser[r.User] = append(byUser[r.User], r)
	}
	users := make([]string, 0, len(byUser))
	for u := range byUser {
		users = append(users, u)
	}
	sort.Slice(users, func(i, j int) bool {
		return qidByUser[users[i]] < qidByUser[users[j]]
	})

	rf, err := os.Create(rankPath)
	if err != nil {
		return Result{}, err
	}
	defer rf.Close()
	w := bufio.NewWriter(rf)
	if _, err := w.WriteString("# qid label feat1 feat2 ...\n"); err != nil {
		return Result{}, err
	}

	var manifest []recsys.ManifestRow
	for _, user := range users {
		qid, ok := qidByUser[user]
		if !ok {
			return Result{}, fmt.Errorf("rankconv: no qid for user %s", user)
		}
		rows := byUser[user]
		sort.Slice(rows, func(i, j int) bool {
			return rows[i].RecallScore > rows[j].RecallScore
		})
		for _, r := range rows {
			label := labelMap[user+"\x00"+r.Item]
			line, err := formatRankLine(qid, label, r.Feats)
			if err != nil {
				return Result{}, err
			}
			if _, err := w.WriteString(line + "\n"); err != nil {
				return Result{}, err
			}
			manifest = append(manifest, recsys.ManifestRow{
				User: user, Item: r.Item, Tag: r.Tag, RecallScore: r.RecallScore,
			})
		}
	}
	if err := w.Flush(); err != nil {
		return Result{}, err
	}
	if err := writeManifest(manifestPath, manifest); err != nil {
		return Result{}, err
	}
	return Result{Manifest: manifest, Rows: len(manifest)}, nil
}

func interactionLabels(samples []recsys.Interaction) map[string]float64 {
	m := map[string]float64{}
	for _, r := range samples {
		key := r.User + "\x00" + r.Item
		if prev, ok := m[key]; !ok || r.Score > prev {
			m[key] = r.Score
		}
	}
	return m
}

func formatRankLine(qid int, label float64, feats []float64) (string, error) {
	var b strings.Builder
	b.WriteString(strconv.Itoa(qid))
	b.WriteByte('\t')
	b.WriteString(strconv.FormatFloat(label, 'g', -1, 64))
	for _, v := range feats {
		b.WriteByte('\t')
		b.WriteString(strconv.FormatFloat(v, 'g', -1, 64))
	}
	return b.String(), nil
}

func writeManifest(path string, rows []recsys.ManifestRow) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	w := bufio.NewWriter(f)
	for _, r := range rows {
		line := fmt.Sprintf("{\"User\":%q,\"Item\":%q,\"Tag\":%q,\"recall_score\":%.6g}\n",
			r.User, r.Item, r.Tag, r.RecallScore)
		if _, err := w.WriteString(line); err != nil {
			return err
		}
	}
	return w.Flush()
}
