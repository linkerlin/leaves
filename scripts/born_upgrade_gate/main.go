package main

// born 升级/复测门禁：真实模型上 Native vs BornCPU/BornGPU 计时对比。
// 用法：go run ./scripts/born_upgrade_gate <model-path>（LEAVES_BORN_GPU=0 跳过 GPU 计时）。
// born 升级或决策表调整前必跑；数字写入 docs/benchmark-baseline.md §再测量。

import (
	"fmt"
	"os"

	_ "github.com/linkerlin/leaves/v2" // 注册 loader
	"github.com/linkerlin/leaves/v2/io"
	"github.com/linkerlin/leaves/v2/tree"
)

func main() {
	modelPath := os.Args[1]
	ens, err := io.LoadFromFile(modelPath, &io.LoadOptions{Backend: io.BackendNative})
	if err != nil {
		panic(err)
	}
	f := ens.Forest()
	fmt.Printf("model=%s trees=%d features=%d nodes=%d\n",
		modelPath, len(f.Trees), f.NumFeatures, countNodes(f))

	caps := tree.ModelCapsFromForest(f, false, true)
	nFeat := f.NumFeatures
	for _, batch := range []int{64, 256, 1024} {
		vals := make([]float64, batch*nFeat)
		for i := range vals {
			vals[i] = float64(i%97) / 97.0
		}
		res := tree.ProfileBackend(caps, vals, batch, nFeat, 20)
		sp := 0.0
		if res.BornCPU.Ok && res.BornCPU.NsPerOp > 0 {
			sp = res.Native.NsPerOp / res.BornCPU.NsPerOp
		}
		gpu := "off"
		if res.BornGPU.Ok && res.BornGPU.NsPerOp > 0 {
			gpu = fmt.Sprintf("%.2fx", res.Native.NsPerOp/res.BornGPU.NsPerOp)
		}
		fmt.Printf("batch=%-5d native=%9.0f born_cpu=%9.0f (%.2fx) born_gpu=%s pick=%s\n",
			batch, res.Native.NsPerOp, res.BornCPU.NsPerOp, sp, gpu, tree.BackendName(res.Pick))
	}
}

func countNodes(f *tree.ForestIR) int {
	n := 0
	for i := range f.Trees {
		n += f.Trees[i].NumNodes
	}
	return n
}
