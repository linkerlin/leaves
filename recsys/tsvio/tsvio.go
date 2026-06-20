package tsvio

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/linkerlin/leaves/recsys"
)

func writeLines(path string, lines []string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	w := bufio.NewWriter(f)
	for _, ln := range lines {
		if _, err := w.WriteString(ln + "\n"); err != nil {
			return err
		}
	}
	return w.Flush()
}

func readDataLines(path string) ([]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var lines []string
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		lines = append(lines, line)
	}
	return lines, sc.Err()
}

// WriteInteractions 写四元 samples TSV。
func WriteInteractions(path string, rows []recsys.Interaction) error {
	lines := []string{"User\tItem\tScore\tTag"}
	for _, r := range rows {
		lines = append(lines, fmt.Sprintf("%s\t%s\t%.6g\t%s", r.User, r.Item, r.Score, r.Tag))
	}
	return writeLines(path, lines)
}

// ReadInteractions 读四元 samples TSV。
func ReadInteractions(path string) ([]recsys.Interaction, error) {
	raw, err := readDataLines(path)
	if err != nil {
		return nil, err
	}
	out := make([]recsys.Interaction, 0, len(raw))
	for i, line := range raw {
		if i == 0 && strings.HasPrefix(line, "User\t") {
			continue
		}
		parts := strings.Split(line, "\t")
		if len(parts) < 4 {
			return nil, fmt.Errorf("tsvio: bad interaction row %d", i+1)
		}
		score, err := strconv.ParseFloat(strings.TrimSpace(parts[2]), 64)
		if err != nil {
			return nil, fmt.Errorf("tsvio: bad score row %d: %w", i+1, err)
		}
		out = append(out, recsys.Interaction{
			User:  parts[0],
			Item:  parts[1],
			Score: score,
			Tag:   parts[3],
		})
	}
	return out, nil
}

// WriteCatalog 写物品目录 TSV。
func WriteCatalog(path string, featNames []string, items []recsys.CatalogItem) error {
	header := "Item\tTag"
	for _, n := range featNames {
		header += "\t" + n
	}
	lines := []string{header}
	for _, it := range items {
		var b strings.Builder
		b.WriteString(it.Item)
		b.WriteByte('\t')
		b.WriteString(it.Tag)
		for _, v := range it.Feats {
			b.WriteByte('\t')
			b.WriteString(strconv.FormatFloat(v, 'g', -1, 64))
		}
		lines = append(lines, b.String())
	}
	return writeLines(path, lines)
}

// ReadCatalog 读物品目录。
func ReadCatalog(path string) (featNames []string, items []recsys.CatalogItem, err error) {
	raw, err := readDataLines(path)
	if err != nil {
		return nil, nil, err
	}
	if len(raw) == 0 {
		return nil, nil, fmt.Errorf("tsvio: empty catalog")
	}
	header := strings.Split(raw[0], "\t")
	if len(header) < 3 || header[0] != "Item" || header[1] != "Tag" {
		return nil, nil, fmt.Errorf("tsvio: bad catalog header")
	}
	featNames = header[2:]
	for _, line := range raw[1:] {
		parts := strings.Split(line, "\t")
		if len(parts) != len(header) {
			return nil, nil, fmt.Errorf("tsvio: catalog column mismatch for %s", parts[0])
		}
		feats := make([]float64, len(featNames))
		for i := range feats {
			feats[i], err = strconv.ParseFloat(parts[2+i], 64)
			if err != nil {
				return nil, nil, fmt.Errorf("tsvio: bad feat %s: %w", parts[0], err)
			}
		}
		items = append(items, recsys.CatalogItem{Item: parts[0], Tag: parts[1], Feats: feats})
	}
	return featNames, items, nil
}

// WriteUserQIDs 写 user_qid 映射。
func WriteUserQIDs(path string, rows []recsys.UserQID) error {
	lines := []string{"User\tqid\tsplit"}
	for _, r := range rows {
		lines = append(lines, fmt.Sprintf("%s\t%d\t%s", r.User, r.QID, r.Split))
	}
	return writeLines(path, lines)
}

// ReadUserQIDs 读 user_qid 映射。
func ReadUserQIDs(path string) ([]recsys.UserQID, error) {
	raw, err := readDataLines(path)
	if err != nil {
		return nil, err
	}
	var out []recsys.UserQID
	for i, line := range raw {
		if i == 0 && strings.HasPrefix(line, "User\t") {
			continue
		}
		parts := strings.Split(line, "\t")
		if len(parts) < 3 {
			return nil, fmt.Errorf("tsvio: bad user_qid row")
		}
		qid, err := strconv.Atoi(parts[1])
		if err != nil {
			return nil, err
		}
		out = append(out, recsys.UserQID{User: parts[0], QID: qid, Split: parts[2]})
	}
	return out, nil
}

// WriteRecall 写召回 TSV。
func WriteRecall(path string, featNames []string, rows []recsys.RecallRow) error {
	header := "User\tItem\tTag\trecall_score"
	for _, n := range featNames {
		header += "\t" + n
	}
	lines := []string{header}
	for _, r := range rows {
		var b strings.Builder
		b.WriteString(r.User)
		b.WriteByte('\t')
		b.WriteString(r.Item)
		b.WriteByte('\t')
		b.WriteString(r.Tag)
		b.WriteByte('\t')
		b.WriteString(strconv.FormatFloat(r.RecallScore, 'g', -1, 64))
		for _, v := range r.Feats {
			b.WriteByte('\t')
			b.WriteString(strconv.FormatFloat(v, 'g', -1, 64))
		}
		lines = append(lines, b.String())
	}
	return writeLines(path, lines)
}

// ReadRecall 读召回 TSV。
func ReadRecall(path string) (featNames []string, rows []recsys.RecallRow, err error) {
	raw, err := readDataLines(path)
	if err != nil {
		return nil, nil, err
	}
	if len(raw) == 0 {
		return nil, nil, fmt.Errorf("tsvio: empty recall")
	}
	header := strings.Split(raw[0], "\t")
	if len(header) < 5 {
		return nil, nil, fmt.Errorf("tsvio: bad recall header")
	}
	featNames = header[4:]
	for _, line := range raw[1:] {
		parts := strings.Split(line, "\t")
		if len(parts) != len(header) {
			return nil, nil, fmt.Errorf("tsvio: recall column mismatch")
		}
		rs, err := strconv.ParseFloat(parts[3], 64)
		if err != nil {
			return nil, nil, err
		}
		feats := make([]float64, len(featNames))
		for i := range feats {
			feats[i], err = strconv.ParseFloat(parts[4+i], 64)
			if err != nil {
				return nil, nil, err
			}
		}
		rows = append(rows, recsys.RecallRow{
			User: parts[0], Item: parts[1], Tag: parts[2],
			RecallScore: rs, Feats: feats,
		})
	}
	return featNames, rows, nil
}

// WriteManifestJSONL 写 manifest。
func WriteManifestJSONL(path string, rows []recsys.ManifestRow) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	w := bufio.NewWriter(f)
	enc := json.NewEncoder(w)
	for _, r := range rows {
		if err := enc.Encode(r); err != nil {
			return err
		}
	}
	return w.Flush()
}

// ReadManifestJSONL 读 manifest。
func ReadManifestJSONL(path string) ([]recsys.ManifestRow, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var out []recsys.ManifestRow
	dec := json.NewDecoder(f)
	for dec.More() {
		var r recsys.ManifestRow
		if err := dec.Decode(&r); err != nil {
			return nil, err
		}
		out = append(out, r)
	}
	return out, nil
}

// WriteDeal 写发牌终稿。
func WriteDeal(path string, rows []recsys.DealRow) error {
	lines := []string{"User\tItem\tTag\tScore\trank"}
	for _, r := range rows {
		lines = append(lines, fmt.Sprintf("%s\t%s\t%s\t%.6g\t%d", r.User, r.Item, r.Tag, r.Score, r.Rank))
	}
	return writeLines(path, lines)
}

// WriteJSON 写 JSON 文件。
func WriteJSON(path string, v any) error {
	b, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, append(b, '\n'), 0o644)
}
