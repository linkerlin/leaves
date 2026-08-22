//go:build js

// GPU-O3：WASM 环境 Native vs BornCPU 实测 harness（goos=js 构建）。
// 运行：GOOS=js GOARCH=wasm go build -o /tmp/bench.wasm ./scripts/wasm_backend_bench
// 然后 node（带 wasm_exec.js）运行，结果 console.log 输出。
package main

import (
	"fmt"
	"math"
	"runtime"
	"time"

	"github.com/linkerlin/leaves/v2/tree"
)

func main() {
	// 合成森林：50 树 × 31 节点（对齐 lg_breast_cancer 规模量级），数值分裂。
	nFeat := 30
	nTrees := 50
	f := &tree.ForestIR{
		NumFeatures:     nFeat,
		NumOutputGroups: 1,
		IterationIndptr: []int{0},
		Name:            "wasm.bench",
	}
	lcg := uint64(42)
	next := func() float64 {
		lcg = lcg*6364136223846793005 + 1442695040888963407
		return float64(lcg>>11) / (1 << 53)
	}
	for t := 0; t < nTrees; t++ {
		const nn = 31
		tir := tree.TreeIR{
			NumNodes:       nn,
			NumLeaves:      nn + 1,
			SplitFeature:   make([]int32, nn),
			SplitThreshold: make([]float64, nn),
			DefaultLeft:    make([]bool, nn),
			MissingZero:    make([]bool, nn),
			MissingNan:     make([]bool, nn),
			LeftChild:      make([]int32, nn),
			RightChild:     make([]int32, nn),
			LeafValue:      make([]float64, nn+1),
			OutputDim:      1,
			MaxDepth:       5,
		}
		leaf := 0
		child := func(idx int) int32 {
			if idx < nn {
				return int32(idx)
			}
			v := int32(^leaf)
			leaf++
			return v
		}
		for i := 0; i < nn; i++ {
			tir.SplitFeature[i] = int32(int(next()*float64(nFeat)) % nFeat)
			tir.SplitThreshold[i] = next()
			tir.DefaultLeft[i] = next() < 0.5
			tir.MissingNan[i] = true
			tir.LeftChild[i] = child(2*i + 1)
			tir.RightChild[i] = child(2*i + 2)
		}
		for i := range tir.LeafValue {
			tir.LeafValue[i] = next()
		}
		f.Trees = append(f.Trees, tir)
		f.WeightDrop = append(f.WeightDrop, 1)
		f.TreeInfo = append(f.TreeInfo, 0)
		f.IterationIndptr = append(f.IterationIndptr, len(f.Trees))
	}

	caps := tree.ModelCapsFromForest(f, false, true)
	fmt.Printf("wasm bench: %d trees x %d nodes, %d features, GOOS=%s\n", nTrees, 31, nFeat, runtime.GOOS)

	for _, batch := range []int{8, 64, 256} {
		vals := make([]float64, batch*nFeat)
		for i := range vals {
			vals[i] = next()
		}
		res := tree.ProfileBackend(caps, vals, batch, nFeat, 10)
		nat, cpu := "n/a", "n/a"
		if res.Native.Ok {
			nat = fmt.Sprintf("%.0fns", res.Native.NsPerOp)
		}
		if res.BornCPU.Ok {
			cpu = fmt.Sprintf("%.0fns (%.2fx)", res.BornCPU.NsPerOp, res.Native.NsPerOp/res.BornCPU.NsPerOp)
		}
		fmt.Printf("batch=%d native=%s born_cpu=%s pick=%s\n", batch, nat, cpu, tree.BackendName(res.Pick))
	}
	_ = math.Pi
	_ = time.Now
}
