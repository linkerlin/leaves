//go:build !js

package tree

import (
	"math"

	"github.com/born-ml/born/tensor"
)

// treeNeedsBornWalk 含 CatSmall / full bitset 分类时须走标量 walkTree。
func treeNeedsBornWalk(t *TreeIR) bool {
	if t == nil {
		return true
	}
	for i := 0; i < t.NumNodes; i++ {
		if i < len(t.IsCategorical) && t.IsCategorical[i] {
			if i >= len(t.CatOneHot) || !t.CatOneHot[i] {
				return true
			}
		}
	}
	return false
}

// walkTreeBatch Born 张量批量树遍历，返回叶编码（负 int32）。
func walkTreeBatch[B tensor.Backend](b B, features *tensor.Tensor[float64, B], t *TreeIR) *tensor.Tensor[int32, B] {
	if t == nil || t.NumNodes == 0 {
		batch := features.Shape()[0]
		return tensor.Zeros[int32](tensor.Shape{batch}, b)
	}
	if treeNeedsBornWalk(t) {
		panic("tree: walkTreeBatch on tree requiring scalar walk")
	}

	batch := features.Shape()[0]
	current := tensor.Zeros[int32](tensor.Shape{batch}, b)

	splitFeat := bornMustFromI32(b, t.SplitFeature[:t.NumNodes], tensor.Shape{t.NumNodes})
	thresholds := bornMustFromF64(b, t.SplitThreshold[:t.NumNodes], tensor.Shape{t.NumNodes})
	leftChild := bornMustFromI32(b, t.LeftChild[:t.NumNodes], tensor.Shape{t.NumNodes})
	rightChild := bornMustFromI32(b, t.RightChild[:t.NumNodes], tensor.Shape{t.NumNodes})

	defaultLeft := bornBoolSliceToF64(b, t.DefaultLeft, t.NumNodes)
	missingZero := bornBoolSliceToF64(b, t.MissingZero, t.NumNodes)
	missingNan := bornBoolSliceToF64(b, t.MissingNan, t.NumNodes)
	isCat := bornBoolSliceToF64(b, t.IsCategorical, t.NumNodes)

	depthLimit := t.MaxDepth
	if depthLimit <= 0 {
		depthLimit = 64
	}

	for depth := 0; depth < depthLimit; depth++ {
		cur := current.Data()
		atLeaf := make([]bool, batch)
		for i, n := range cur {
			atLeaf[i] = n < 0
		}
		active := bornMaskNodeIndices(cur, b)

		featIdx := splitFeat.Gather(0, active)
		featVal := bornGatherFeatureValues(features, featIdx)

		thr := thresholds.Gather(0, active)
		dLeft := defaultLeft.Gather(0, active)
		mZero := missingZero.Gather(0, active)
		mNan := missingNan.Gather(0, active)
		catN := isCat.Gather(0, active)

		goLeft := bornBatchGoLeft(featVal, thr, dLeft, mZero, mNan, catN, batch)

		left := leftChild.Gather(0, active)
		right := rightChild.Gather(0, active)
		current = bornMergeWalkStep(cur, atLeaf, goLeft, left.Data(), right.Data(), b)

		if bornAllAtLeaf(current.Data()) {
			break
		}
	}
	return current
}

func bornAllAtLeaf(nodes []int32) bool {
	for _, n := range nodes {
		if n >= 0 {
			return false
		}
	}
	return true
}

func bornBoolSliceToF64[B tensor.Backend](b B, flags []bool, n int) *tensor.Tensor[float64, B] {
	data := make([]float64, n)
	for i := 0; i < n && i < len(flags); i++ {
		if flags[i] {
			data[i] = 1
		}
	}
	return bornMustFromF64(b, data, tensor.Shape{n})
}

func bornBatchGoLeft[B tensor.Backend](
	featVal, thr, dLeft, mZero, mNan, isCat *tensor.Tensor[float64, B],
	batch int,
) []bool {
	fv := featVal.Data()
	th := thr.Data()
	dl := dLeft.Data()
	mz := mZero.Data()
	mn := mNan.Data()
	ic := isCat.Data()
	out := make([]bool, batch)
	for i := 0; i < batch; i++ {
		f := fv[i]
		missing := false
		if i < len(mz) && mz[i] > 0.5 && isZeroFval(f) {
			missing = true
		}
		if i < len(mn) && mn[i] > 0.5 && math.IsNaN(f) {
			missing = true
		}
		if missing {
			out[i] = i < len(dl) && dl[i] > 0.5
			continue
		}
		t := 0.0
		if i < len(th) {
			t = th[i]
		}
		if i < len(ic) && ic[i] > 0.5 {
			out[i] = f == t
		} else {
			if math.IsNaN(f) {
				f = 0
			}
			out[i] = f <= t
		}
	}
	return out
}

func bornMaskNodeIndices[B tensor.Backend](nodes []int32, b B) *tensor.Tensor[int32, B] {
	masked := make([]int32, len(nodes))
	for i, n := range nodes {
		if n < 0 {
			masked[i] = 0
		} else {
			masked[i] = n
		}
	}
	return bornMustFromI32(b, masked, tensor.Shape{len(nodes)})
}

func bornMergeWalkStep[B tensor.Backend](
	cur []int32,
	atLeaf []bool,
	goLeft []bool,
	left, right []int32,
	b B,
) *tensor.Tensor[int32, B] {
	batch := len(cur)
	next := make([]int32, batch)
	for i := 0; i < batch; i++ {
		if atLeaf[i] {
			next[i] = cur[i]
			continue
		}
		if goLeft[i] {
			next[i] = left[i]
		} else {
			next[i] = right[i]
		}
	}
	t, err := tensor.FromSlice(next, tensor.Shape{batch}, b)
	if err != nil {
		panic(err)
	}
	return t
}

func treeScalarBatch[B tensor.Backend](b B, features *tensor.Tensor[float64, B], t *TreeIR) []float64 {
	batch := features.Shape()[0]
	if t.NumNodes == 0 {
		out := make([]float64, batch)
		v := 0.0
		if len(t.LeafValue) > 0 {
			v = t.LeafValue[0]
		}
		for i := range out {
			out[i] = v
		}
		return out
	}
	leafNode := walkTreeBatch(b, features, t)
	out := make([]float64, batch)
	for i, n := range leafNode.Data() {
		out[i] = treeLeafScalar(t, n)
	}
	return out
}

func treeVectorBatch[B tensor.Backend](b B, features *tensor.Tensor[float64, B], t *TreeIR) [][]float64 {
	batch := features.Shape()[0]
	dim := t.OutputDim
	if dim <= 0 {
		dim = 1
	}
	out := make([][]float64, batch)
	for i := range out {
		out[i] = make([]float64, dim)
	}
	if t.NumNodes == 0 {
		for i := 0; i < batch; i++ {
			for d := 0; d < dim && d < len(t.LeafValue); d++ {
				out[i][d] = t.LeafValue[d]
			}
		}
		return out
	}
	leafNode := walkTreeBatch(b, features, t)
	for i, n := range leafNode.Data() {
		copy(out[i], treeLeafVector(t, n))
	}
	return out
}
