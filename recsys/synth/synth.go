package synth

import (
	"fmt"
	"math"
	"math/rand"
	"sort"

	"github.com/linkerlin/leaves/recsys"
)

var defaultTags = []string{"drama", "comedy", "action", "doc"}

// FeatNames smoke 用 4 维特征列名。
var FeatNames = []string{"feat_pop", "feat_quality", "feat_age", "feat_bias"}

// Dataset 合成数据集：原始交互 + 物品目录。
type Dataset struct {
	Raw         []recsys.Interaction
	Catalog     []recsys.CatalogItem
	FeatNames   []string
	TrainUsers  []string
	TestUsers   []string
}

// Generate 合成推荐 smoke 原始数据。
func Generate(cfg recsys.SmokeConfig) (Dataset, error) {
	if cfg.TrainUsers <= 0 || cfg.TestUsers <= 0 {
		return Dataset{}, fmt.Errorf("synth: need positive train/test users")
	}
	if cfg.NumItems < cfg.RecallSize {
		return Dataset{}, fmt.Errorf("synth: NumItems %d < RecallSize %d", cfg.NumItems, cfg.RecallSize)
	}
	rng := rand.New(rand.NewSource(cfg.Seed))

	catalog := buildCatalog(rng, cfg.NumItems, defaultTags)
	userTags := assignUserTags(rng, cfg.TrainUsers+cfg.TestUsers, defaultTags)

	allUsers := make([]string, 0, cfg.TrainUsers+cfg.TestUsers)
	for i := 0; i < cfg.TrainUsers+cfg.TestUsers; i++ {
		allUsers = append(allUsers, fmt.Sprintf("u%03d", i))
	}
	trainUsers := allUsers[:cfg.TrainUsers]
	testUsers := allUsers[cfg.TrainUsers:]

	itemByID := indexCatalog(catalog)
	var raw []recsys.Interaction
	for ui, user := range allUsers {
		prefTag := userTags[ui]
		nEvents := cfg.MinEvents + rng.Intn(8)
		seen := map[string]struct{}{}
		for e := 0; e < nEvents; e++ {
			var it recsys.CatalogItem
			if rng.Float64() < 0.7 {
				it = pickItemByTag(rng, catalog, prefTag)
			} else {
				it = catalog[rng.Intn(len(catalog))]
			}
			if _, ok := seen[it.Item]; ok {
				continue
			}
			seen[it.Item] = struct{}{}
			score := labelForItem(rng, it, prefTag)
			raw = append(raw, recsys.Interaction{
				User: user, Item: it.Item, Tag: it.Tag, Score: score,
			})
		}
		_ = itemByID
	}

	return Dataset{
		Raw:        raw,
		Catalog:    catalog,
		FeatNames:  FeatNames,
		TrainUsers: trainUsers,
		TestUsers:  testUsers,
	}, nil
}

func buildCatalog(rng *rand.Rand, n int, tags []string) []recsys.CatalogItem {
	items := make([]recsys.CatalogItem, n)
	for i := 0; i < n; i++ {
		tag := tags[i%len(tags)]
		pop := math.Log1p(float64(rng.Intn(200) + 1))
		quality := rng.Float64()
		age := rng.Float64()
		bias := rng.NormFloat64() * 0.1
		items[i] = recsys.CatalogItem{
			Item:  fmt.Sprintf("i%04d", i),
			Tag:   tag,
			Feats: []float64{pop, quality, age, bias},
		}
	}
	return items
}

func assignUserTags(rng *rand.Rand, n int, tags []string) []string {
	out := make([]string, n)
	for i := range out {
		out[i] = tags[rng.Intn(len(tags))]
	}
	return out
}

func indexCatalog(cat []recsys.CatalogItem) map[string]recsys.CatalogItem {
	m := make(map[string]recsys.CatalogItem, len(cat))
	for _, it := range cat {
		m[it.Item] = it
	}
	return m
}

func pickItemByTag(rng *rand.Rand, catalog []recsys.CatalogItem, tag string) recsys.CatalogItem {
	var pool []recsys.CatalogItem
	for _, it := range catalog {
		if it.Tag == tag {
			pool = append(pool, it)
		}
	}
	if len(pool) == 0 {
		return catalog[rng.Intn(len(catalog))]
	}
	return pool[rng.Intn(len(pool))]
}

func labelForItem(rng *rand.Rand, it recsys.CatalogItem, prefTag string) float64 {
	base := it.Feats[1] * 4
	if it.Tag == prefTag {
		base += 1.5
	}
	base += rng.NormFloat64() * 0.3
	if base < 0 {
		return 0
	}
	if base > 4 {
		return 4
	}
	return math.Round(base)
}

// SplitInteractions 按用户列表切分交互。
func SplitInteractions(all []recsys.Interaction, trainUsers, testUsers []string) (train, test []recsys.Interaction) {
	trainSet := toSet(trainUsers)
	testSet := toSet(testUsers)
	for _, r := range all {
		if _, ok := trainSet[r.User]; ok {
			train = append(train, r)
		} else if _, ok := testSet[r.User]; ok {
			test = append(test, r)
		}
	}
	return train, test
}

func toSet(users []string) map[string]struct{} {
	m := make(map[string]struct{}, len(users))
	for _, u := range users {
		m[u] = struct{}{}
	}
	return m
}

// TagVocab 从目录提取 Tag 词表。
func TagVocab(catalog []recsys.CatalogItem) []string {
	set := map[string]struct{}{}
	for _, it := range catalog {
		set[it.Tag] = struct{}{}
	}
	out := make([]string, 0, len(set))
	for t := range set {
		out = append(out, t)
	}
	sort.Strings(out)
	return out
}
