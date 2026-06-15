package tree

import (
	"fmt"
	"math"
)

// ApplyXGBCategoricalSplits 将 XGBoost JSON 分类分裂元数据写入 TreeIR。
func ApplyXGBCategoricalSplits(
	t *TreeIR,
	splitType []int,
	categories []int32,
	categoriesNodes []int32,
	categoriesSegments []int64,
	categoriesSizes []int64,
) {
	if t == nil || len(splitType) == 0 || t.NumNodes == 0 {
		return
	}
	nNodes := t.NumNodes
	if len(t.IsCategorical) < nNodes {
		t.IsCategorical = growBoolSlice(t.IsCategorical, nNodes)
	}
	if len(t.CatOneHot) < nNodes {
		t.CatOneHot = growBoolSlice(t.CatOneHot, nNodes)
	}
	if len(t.CatSmall) < nNodes {
		t.CatSmall = growBoolSlice(t.CatSmall, nNodes)
	}

	catCnt := 0
	lastCatNode := int32(-1)
	if len(categoriesNodes) > 0 {
		lastCatNode = categoriesNodes[0]
	}

	var allBits []uint32
	t.CatBoundaries = []uint32{0}

	for nidx := 0; nidx < nNodes && nidx < len(splitType); nidx++ {
		if splitType[nidx] != 1 {
			continue
		}
		t.IsCategorical[nidx] = true
		if int32(nidx) != lastCatNode {
			continue
		}
		if catCnt >= len(categoriesSegments) || catCnt >= len(categoriesSizes) {
			break
		}
		jBegin := int(categoriesSegments[catCnt])
		jEnd := jBegin + int(categoriesSizes[catCnt])
		if jBegin < 0 || jEnd > len(categories) || jEnd <= jBegin {
			catCnt++
			if catCnt < len(categoriesNodes) {
				lastCatNode = categoriesNodes[catCnt]
			} else {
				lastCatNode = -1
			}
			continue
		}

		maxCat := int32(0)
		for j := jBegin; j < jEnd; j++ {
			if categories[j] > maxCat {
				maxCat = categories[j]
			}
		}
		nCats := int(maxCat) + 1
		nWords := (nCats + 31) / 32
		words := make([]uint32, nWords)
		for j := jBegin; j < jEnd; j++ {
			cat := categories[j]
			if cat < 0 {
				continue
			}
			words[cat/32] |= 1 << (cat % 32)
		}

		catIdx := len(t.CatBoundaries) - 1
		t.SplitThreshold[nidx] = float64(catIdx)
		allBits = append(allBits, words...)
		t.CatBoundaries = append(t.CatBoundaries, uint32(len(allBits)))

		catCnt++
		if catCnt < len(categoriesNodes) {
			lastCatNode = categoriesNodes[catCnt]
		} else {
			lastCatNode = -1
		}
	}
	t.CatThresholds = allBits
}

func growBoolSlice(s []bool, n int) []bool {
	if len(s) >= n {
		return s
	}
	out := make([]bool, n)
	copy(out, s)
	return out
}

// ValidateXGBCategoricalNode 用于测试：节点是否为有效 XGB 分类分裂。
func ValidateXGBCategoricalNode(t *TreeIR, nodeIdx int) error {
	if t == nil || nodeIdx < 0 || nodeIdx >= t.NumNodes {
		return fmt.Errorf("invalid node")
	}
	if nodeIdx >= len(t.IsCategorical) || !t.IsCategorical[nodeIdx] {
		return fmt.Errorf("not categorical")
	}
	catIdx := int(t.SplitThreshold[nodeIdx])
	if catIdx+1 >= len(t.CatBoundaries) {
		return fmt.Errorf("missing cat boundaries")
	}
	if t.CatBoundaries[catIdx] >= t.CatBoundaries[catIdx+1] {
		return fmt.Errorf("empty cat bitset")
	}
	return nil
}

// XGBCategoricalGoLeft 判断分类特征值是否命中 XGBoost bitset（走左分支）。
func XGBCategoricalGoLeft(t *TreeIR, nodeIdx int, fval float64) bool {
	if nodeIdx < 0 || nodeIdx >= t.NumNodes {
		return false
	}
	if math.IsNaN(fval) {
		if nodeIdx < len(t.MissingNan) && t.MissingNan[nodeIdx] {
			if nodeIdx < len(t.DefaultLeft) {
				return t.DefaultLeft[nodeIdx]
			}
			return false
		}
		fval = 0
	}
	cat := int32(fval)
	if cat < 0 {
		return false
	}
	catIdx := int(t.SplitThreshold[nodeIdx])
	if catIdx+1 >= len(t.CatBoundaries) {
		return false
	}
	start := t.CatBoundaries[catIdx]
	end := t.CatBoundaries[catIdx+1]
	if start >= end {
		return false
	}
	wordIdx := start + uint32(cat/32)
	if int(wordIdx) >= len(t.CatThresholds) {
		return false
	}
	bit := uint32(cat % 32)
	return (t.CatThresholds[wordIdx]>>bit)&1 > 0
}

// XGBTreeCatExport XGBoost JSON 分类分裂导出字段。
type XGBTreeCatExport struct {
	SplitType          []int
	Categories         []int32
	CategoriesNodes    []int32
	CategoriesSegments []int64
	CategoriesSizes    []int64
}

// ExportXGBTreeCatMeta 从 TreeIR 导出 XGBoost 分类元数据。
func ExportXGBTreeCatMeta(t *TreeIR) XGBTreeCatExport {
	out := XGBTreeCatExport{}
	if t == nil {
		return out
	}
	nNodes := t.NumNodes
	if nNodes <= 0 {
		nNodes = len(t.LeftChild)
	}
	if nNodes <= 0 {
		return out
	}
	out.SplitType = make([]int, nNodes)
	var categories []int32
	for nidx := 0; nidx < nNodes; nidx++ {
		if nidx >= len(t.IsCategorical) || !t.IsCategorical[nidx] {
			continue
		}
		out.SplitType[nidx] = 1
		catIdx := int(t.SplitThreshold[nidx])
		if catIdx+1 >= len(t.CatBoundaries) {
			continue
		}
		start := t.CatBoundaries[catIdx]
		end := t.CatBoundaries[catIdx+1]
		out.CategoriesSegments = append(out.CategoriesSegments, int64(len(categories)))
		var nCats int64
		for wi := start; wi < end && int(wi) < len(t.CatThresholds); wi++ {
			word := t.CatThresholds[wi]
			baseCat := int(wi-start) * 32
			for b := 0; b < 32; b++ {
				if (word>>uint(b))&1 != 0 {
					categories = append(categories, int32(baseCat+b))
					nCats++
				}
			}
		}
		out.CategoriesSizes = append(out.CategoriesSizes, nCats)
		out.CategoriesNodes = append(out.CategoriesNodes, int32(nidx))
	}
	out.Categories = categories
	return out
}
