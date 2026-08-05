// Package movielens 将 MovieLens 100K 转为 recsys 四元 Dataset（User/Item/Score/Tag + catalog）。
package movielens

import (
	"archive/zip"
	"bufio"
	"fmt"
	"io"
	"math"
	"math/rand"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/linkerlin/leaves/v2/recsys"
	"github.com/linkerlin/leaves/v2/recsys/synth"
)

const (
	DefaultURL       = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
	DefaultCacheRel  = ".cache/ml-100k.zip"
	DefaultMinRating = 12
	DefaultTrainU    = 60
	DefaultTestU     = 15
	nGenre           = 19
)

// 与 ml-100k/u.genre 顺序一致。
var genreNames = []string{
	"unknown", "Action", "Adventure", "Animation", "Children",
	"Comedy", "Crime", "Documentary", "Drama", "Fantasy",
	"Film-Noir", "Horror", "Musical", "Mystery", "Romance",
	"Sci-Fi", "Thriller", "War", "Western",
}

// Config MovieLens → Dataset 参数。
type Config struct {
	// CachePath 本地 zip；空则用 <RepoRoot>/.cache/ml-100k.zip 或 cwd 相对路径。
	CachePath  string
	URL        string
	Seed       int64
	TrainUsers int
	TestUsers  int
	MinRatings int
	// RepoRoot 用于解析默认 cache；空则尝试 cwd。
	RepoRoot string
}

// DefaultConfig 与 gen_rank_movielens.py 默认切分规模一致。
func DefaultConfig() Config {
	return Config{
		URL:        DefaultURL,
		Seed:       42,
		TrainUsers: DefaultTrainU,
		TestUsers:  DefaultTestU,
		MinRatings: DefaultMinRating,
	}
}

// Load 下载/读缓存 ml-100k，产出 synth.Dataset 与 Item→Title 映射。
func Load(cfg Config) (synth.Dataset, map[string]string, error) {
	if cfg.TrainUsers <= 0 || cfg.TestUsers <= 0 {
		return synth.Dataset{}, nil, fmt.Errorf("movielens: need positive train/test users")
	}
	if cfg.MinRatings <= 0 {
		cfg.MinRatings = DefaultMinRating
	}
	if cfg.URL == "" {
		cfg.URL = DefaultURL
	}
	zipPath, err := ensureZip(cfg)
	if err != nil {
		return synth.Dataset{}, nil, err
	}
	return loadFromZip(zipPath, cfg)
}

// FeatNames 目录与召回特征列名（与 gen_rank_movielens 语义对齐）。
func FeatNames() []string {
	names := []string{"feat_pop", "feat_avg_rating", "feat_year"}
	for _, g := range genreNames {
		safe := strings.ReplaceAll(strings.ToLower(g), "-", "_")
		names = append(names, "feat_genre_"+safe)
	}
	return names
}

func ensureZip(cfg Config) (string, error) {
	path := cfg.CachePath
	if path == "" {
		root := cfg.RepoRoot
		if root == "" {
			cwd, _ := os.Getwd()
			root = findRepoRoot(cwd)
		}
		if root != "" {
			path = filepath.Join(root, DefaultCacheRel)
		} else {
			path = DefaultCacheRel
		}
	}
	if st, err := os.Stat(path); err == nil && st.Size() > 0 {
		return path, nil
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return "", fmt.Errorf("movielens: mkdir cache: %w", err)
	}
	client := &http.Client{Timeout: 120 * time.Second}
	resp, err := client.Get(cfg.URL)
	if err != nil {
		return "", fmt.Errorf("movielens: download: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("movielens: download status %s", resp.Status)
	}
	f, err := os.Create(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	if _, err := io.Copy(f, resp.Body); err != nil {
		return "", fmt.Errorf("movielens: write cache: %w", err)
	}
	return path, nil
}

func loadFromZip(zipPath string, cfg Config) (synth.Dataset, map[string]string, error) {
	zr, err := zip.OpenReader(zipPath)
	if err != nil {
		return synth.Dataset{}, nil, fmt.Errorf("movielens: open zip: %w", err)
	}
	defer zr.Close()

	type movie struct {
		title  string
		year   float64
		genres []float64
	}
	movies := map[int]movie{}
	titles := map[string]string{}

	itemRC, err := openZipFile(zr, "ml-100k/u.item")
	if err != nil {
		return synth.Dataset{}, nil, err
	}
	sc := bufio.NewScanner(itemRC)
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for sc.Scan() {
		line := sc.Text()
		parts := strings.Split(line, "|")
		if len(parts) < 5+nGenre {
			continue
		}
		mid, err := strconv.Atoi(parts[0])
		if err != nil {
			continue
		}
		title := strings.TrimSpace(parts[1])
		year := parseYear(parts[2])
		genres := make([]float64, nGenre)
		for i := 0; i < nGenre; i++ {
			v, _ := strconv.ParseFloat(parts[5+i], 64)
			genres[i] = v
		}
		movies[mid] = movie{title: title, year: year, genres: genres}
		titles[strconv.Itoa(mid)] = title
	}
	_ = itemRC.Close()
	if err := sc.Err(); err != nil {
		return synth.Dataset{}, nil, err
	}

	type rating struct {
		user, movie int
		score       float64
	}
	var ratings []rating
	pop := map[int]int{}
	sumR := map[int]float64{}

	dataRC, err := openZipFile(zr, "ml-100k/u.data")
	if err != nil {
		return synth.Dataset{}, nil, err
	}
	sc = bufio.NewScanner(dataRC)
	for sc.Scan() {
		parts := strings.Fields(sc.Text())
		if len(parts) < 3 {
			continue
		}
		u, _ := strconv.Atoi(parts[0])
		m, _ := strconv.Atoi(parts[1])
		r, _ := strconv.ParseFloat(parts[2], 64)
		if _, ok := movies[m]; !ok {
			continue
		}
		ratings = append(ratings, rating{user: u, movie: m, score: r})
		pop[m]++
		sumR[m] += r
	}
	_ = dataRC.Close()
	if err := sc.Err(); err != nil {
		return synth.Dataset{}, nil, err
	}

	// catalog
	featNames := FeatNames()
	var catalog []recsys.CatalogItem
	mids := make([]int, 0, len(movies))
	for mid := range movies {
		mids = append(mids, mid)
	}
	sort.Ints(mids)
	for _, mid := range mids {
		mv := movies[mid]
		p := float64(pop[mid])
		avg := 0.0
		if p > 0 {
			avg = sumR[mid] / p
		}
		feats := make([]float64, 0, 3+nGenre)
		feats = append(feats, math.Log1p(p), avg, (mv.year-1970.0)/50.0)
		feats = append(feats, mv.genres...)
		catalog = append(catalog, recsys.CatalogItem{
			Item:  strconv.Itoa(mid),
			Tag:   primaryTag(mv.genres),
			Feats: feats,
		})
	}

	// per-user ratings
	byUser := map[int][]rating{}
	for _, r := range ratings {
		byUser[r.user] = append(byUser[r.user], r)
	}
	var eligible []int
	for u, rows := range byUser {
		if len(rows) >= cfg.MinRatings {
			eligible = append(eligible, u)
		}
	}
	sort.Ints(eligible)
	need := cfg.TrainUsers + cfg.TestUsers
	if len(eligible) < need {
		return synth.Dataset{}, nil, fmt.Errorf("movielens: only %d eligible users, need %d", len(eligible), need)
	}

	rng := rand.New(rand.NewSource(cfg.Seed))
	rng.Shuffle(len(eligible), func(i, j int) { eligible[i], eligible[j] = eligible[j], eligible[i] })
	picked := eligible[:need]
	trainU := picked[:cfg.TrainUsers]
	testU := picked[cfg.TrainUsers:]

	userStr := func(ids []int) []string {
		out := make([]string, len(ids))
		for i, id := range ids {
			out[i] = strconv.Itoa(id)
		}
		return out
	}
	trainUsers := userStr(trainU)
	testUsers := userStr(testU)
	want := map[int]struct{}{}
	for _, u := range trainU {
		want[u] = struct{}{}
	}
	for _, u := range testU {
		want[u] = struct{}{}
	}

	var raw []recsys.Interaction
	for _, r := range ratings {
		if _, ok := want[r.user]; !ok {
			continue
		}
		mv := movies[r.movie]
		raw = append(raw, recsys.Interaction{
			User:  strconv.Itoa(r.user),
			Item:  strconv.Itoa(r.movie),
			Tag:   primaryTag(mv.genres),
			Score: r.score,
		})
	}

	return synth.Dataset{
		Raw:        raw,
		Catalog:    catalog,
		FeatNames:  featNames,
		TrainUsers: trainUsers,
		TestUsers:  testUsers,
	}, titles, nil
}

func primaryTag(genres []float64) string {
	for i, g := range genres {
		if g > 0 && i < len(genreNames) {
			return genreNames[i]
		}
	}
	return "unknown"
}

func parseYear(dateS string) float64 {
	dateS = strings.TrimSpace(dateS)
	if len(dateS) < 4 {
		return 1995
	}
	// formats: "01-Jan-1995" or year at end
	y, err := strconv.ParseFloat(dateS[len(dateS)-4:], 64)
	if err != nil {
		return 1995
	}
	return y
}

func openZipFile(zr *zip.ReadCloser, name string) (io.ReadCloser, error) {
	for _, f := range zr.File {
		if f.Name == name || strings.HasSuffix(f.Name, "/"+filepath.Base(name)) {
			return f.Open()
		}
	}
	// fallback: any path ending with u.item / u.data
	base := filepath.Base(name)
	for _, f := range zr.File {
		if filepath.Base(f.Name) == base {
			return f.Open()
		}
	}
	return nil, fmt.Errorf("movielens: %s not in zip", name)
}

func findRepoRoot(start string) string {
	dir := start
	for i := 0; i < 8; i++ {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			break
		}
		dir = parent
	}
	return ""
}
