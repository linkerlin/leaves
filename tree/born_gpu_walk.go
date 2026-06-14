//go:build windows && !js

package tree

import (
	"fmt"
	"math"

	bornwebgpu "github.com/born-ml/born/backend/webgpu"
	"github.com/born-ml/born/tensor"
)

// bornForestMarginsDenseGPU WebGPU 路径（float32 张量；Gather 不支持 f64）。
func bornForestMarginsDenseGPU(
	b *bornwebgpu.Backend,
	f *ForestIR,
	vals []float64,
	rows, cols, nEst int,
) ([][]float64, error) {
	if f == nil || rows <= 0 {
		return nil, fmt.Errorf("born: invalid forest or rows")
	}
	g := f.NumOutputGroups
	if g <= 0 {
		g = 1
	}
	margins := make([][]float64, rows)
	for i := range margins {
		margins[i] = make([]float64, g)
		for k := 0; k < g; k++ {
			margins[i][k] = classBaseScore(f, k)
		}
	}

	nEstAdj := adjustNEstimators(f, nEst)
	if nEstAdj <= 0 {
		return margins, nil
	}

	valsF32 := f64SliceToF32(vals)
	feats := bornDenseFeaturesF32(b, valsF32, rows, cols)

	addTree := func(ti int, coef float64) {
		if ti < 0 || ti >= len(f.Trees) {
			return
		}
		t := &f.Trees[ti]
		w := weightDrop(f, ti) * coef
		k := forestTreeClassIndex(f, ti)

		if treeNeedsBornWalk(t) {
			for r := 0; r < rows; r++ {
				fv := bornRowSlice(vals, r, cols)
				if t.OutputDim > 1 {
					vec := treeVectorMargin(t, fv)
					for d := 0; d < len(vec) && d < g; d++ {
						margins[r][d] += vec[d] * w
					}
				} else if k >= 0 && k < g {
					margins[r][k] += predictTreeScalar(t, fv) * w
				}
			}
			return
		}

		if t.OutputDim > 1 {
			vecs := treeVectorBatchF32(b, feats, t)
			for r := 0; r < rows; r++ {
				for d := 0; d < len(vecs[r]) && d < g; d++ {
					margins[r][d] += vecs[r][d] * w
				}
			}
			return
		}
		scalars := treeScalarBatchF32(b, feats, t)
		for r := 0; r < rows; r++ {
			if k >= 0 && k < g {
				margins[r][k] += scalars[r] * w
			}
		}
	}

	if len(f.IterationIndptr) > 1 {
		for iter := 0; iter < nEstAdj; iter++ {
			if iter+1 >= len(f.IterationIndptr) {
				break
			}
			for ti := f.IterationIndptr[iter]; ti < f.IterationIndptr[iter+1]; ti++ {
				addTree(ti, 1.0)
			}
		}
		if f.AverageOutput {
			bornScaleMarginsExceptBase(f, margins, 1.0/float64(nEstAdj))
		}
		return margins, nil
	}

	coef := 1.0
	if f.AverageOutput {
		coef = 1.0 / float64(nEstAdj)
	}
	for i := 0; i < nEstAdj; i++ {
		for k := 0; k < g; k++ {
			addTree(i*g+k, coef)
		}
	}
	return margins, nil
}

func f64SliceToF32(in []float64) []float32 {
	out := make([]float32, len(in))
	for i, v := range in {
		out[i] = float32(v)
	}
	return out
}

func bornDenseFeaturesF32(b *bornwebgpu.Backend, vals []float32, rows, cols int) *tensor.Tensor[float32, *bornwebgpu.Backend] {
	t, err := tensor.FromSlice(vals, tensor.Shape{rows, cols}, b)
	if err != nil {
		panic(err)
	}
	return t
}

func walkTreeBatchF32(b *bornwebgpu.Backend, features *tensor.Tensor[float32, *bornwebgpu.Backend], t *TreeIR) *tensor.Tensor[int32, *bornwebgpu.Backend] {
	if t == nil || t.NumNodes == 0 {
		batch := features.Shape()[0]
		return tensor.Zeros[int32](tensor.Shape{batch}, b)
	}
	if treeNeedsBornWalk(t) {
		panic("tree: walkTreeBatchF32 on tree requiring scalar walk")
	}

	batch := features.Shape()[0]
	current := tensor.Zeros[int32](tensor.Shape{batch}, b)

	splitFeat := i32SliceToF32Tensor(b, t.SplitFeature[:t.NumNodes])
	thresholds := f64NodesToF32Tensor(b, t.SplitThreshold[:t.NumNodes])
	leftChild := i32SliceToF32Tensor(b, t.LeftChild[:t.NumNodes])
	rightChild := i32SliceToF32Tensor(b, t.RightChild[:t.NumNodes])
	defaultLeft := boolSliceToF32(b, t.DefaultLeft, t.NumNodes)
	missingZero := boolSliceToF32(b, t.MissingZero, t.NumNodes)
	missingNan := boolSliceToF32(b, t.MissingNan, t.NumNodes)
	isCat := boolSliceToF32(b, t.IsCategorical, t.NumNodes)

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

		featIdxF32 := splitFeat.Gather(0, active)
		featIdx := f32GatherToI32(b, featIdxF32)
		featVal := features.Gather(1, featIdx.Unsqueeze(1))

		thr := thresholds.Gather(0, active)
		dLeft := defaultLeft.Gather(0, active)
		mZero := missingZero.Gather(0, active)
		mNan := missingNan.Gather(0, active)
		catN := isCat.Gather(0, active)

		goLeft := bornBatchGoLeftF32(featVal, thr, dLeft, mZero, mNan, catN, batch)

		leftF32 := leftChild.Gather(0, active)
		rightF32 := rightChild.Gather(0, active)
		left := f32DataToI32(leftF32.Data())
		right := f32DataToI32(rightF32.Data())
		current = bornMergeWalkStep(cur, atLeaf, goLeft, left, right, b)

		if bornAllAtLeaf(current.Data()) {
			break
		}
	}
	return current
}

func i32SliceToF32Tensor(b *bornwebgpu.Backend, nodes []int32) *tensor.Tensor[float32, *bornwebgpu.Backend] {
	data := make([]float32, len(nodes))
	for i, v := range nodes {
		data[i] = float32(v)
	}
	t, err := tensor.FromSlice(data, tensor.Shape{len(nodes)}, b)
	if err != nil {
		panic(err)
	}
	return t
}

func f32GatherToI32(b *bornwebgpu.Backend, t *tensor.Tensor[float32, *bornwebgpu.Backend]) *tensor.Tensor[int32, *bornwebgpu.Backend] {
	data := f32DataToI32(t.Data())
	out, err := tensor.FromSlice(data, t.Shape(), b)
	if err != nil {
		panic(err)
	}
	return out
}

func f32DataToI32(data []float32) []int32 {
	out := make([]int32, len(data))
	for i, v := range data {
		out[i] = int32(math.Round(float64(v)))
	}
	return out
}

func f64NodesToF32Tensor(b *bornwebgpu.Backend, nodes []float64) *tensor.Tensor[float32, *bornwebgpu.Backend] {
	data := make([]float32, len(nodes))
	for i, v := range nodes {
		data[i] = float32(v)
	}
	t, err := tensor.FromSlice(data, tensor.Shape{len(nodes)}, b)
	if err != nil {
		panic(err)
	}
	return t
}

func boolSliceToF32(b *bornwebgpu.Backend, flags []bool, n int) *tensor.Tensor[float32, *bornwebgpu.Backend] {
	data := make([]float32, n)
	for i := 0; i < n && i < len(flags); i++ {
		if flags[i] {
			data[i] = 1
		}
	}
	t, err := tensor.FromSlice(data, tensor.Shape{n}, b)
	if err != nil {
		panic(err)
	}
	return t
}

func bornBatchGoLeftF32(
	featVal, thr, dLeft, mZero, mNan, isCat *tensor.Tensor[float32, *bornwebgpu.Backend],
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
		f := float64(fv[i])
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
			t = float64(th[i])
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

func treeScalarBatchF32(b *bornwebgpu.Backend, features *tensor.Tensor[float32, *bornwebgpu.Backend], t *TreeIR) []float64 {
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
	leafNode := walkTreeBatchF32(b, features, t)
	out := make([]float64, batch)
	for i, n := range leafNode.Data() {
		out[i] = treeLeafScalar(t, n)
	}
	return out
}

func treeVectorBatchF32(b *bornwebgpu.Backend, features *tensor.Tensor[float32, *bornwebgpu.Backend], t *TreeIR) [][]float64 {
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
	leafNode := walkTreeBatchF32(b, features, t)
	for i, n := range leafNode.Data() {
		copy(out[i], treeLeafVector(t, n))
	}
	return out
}
