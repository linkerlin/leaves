//go:build windows && !js

// wgpu_repro 是 GPU-O / REV-10 的最小化 WebGPU 复现：
// 反复 FromSlice + Gather + .Data() 往返。
// 参考机（wgpu v0.30.x）上曾出现 NsPerOp≈0、batch 大时挂起、vkMapMemory panic。
//
//	go run ./scripts/wgpu_repro
//	LEAVES_BORN_GPU=0 go run ./scripts/wgpu_repro   # 应 exit 2
package main

import (
	"fmt"
	"os"
	"strings"
	"time"

	bornwebgpu "github.com/born-ml/born/backend/webgpu"
	"github.com/born-ml/born/tensor"
)

func main() {
	switch strings.ToLower(strings.TrimSpace(os.Getenv("LEAVES_BORN_GPU"))) {
	case "0", "off", "false":
		fmt.Fprintln(os.Stderr, "LEAVES_BORN_GPU disabled")
		os.Exit(2)
	}
	if !bornwebgpu.IsAvailable() {
		fmt.Fprintln(os.Stderr, "born webgpu unavailable")
		os.Exit(2)
	}
	b, err := bornwebgpu.New()
	if err != nil {
		fmt.Fprintf(os.Stderr, "bornwebgpu.New: %v\n", err)
		os.Exit(2)
	}
	defer b.Release()

	const n, rounds = 64, 80
	vals := make([]float32, n)
	idx := make([]int32, n)
	for i := range vals {
		vals[i] = float32(i)
		idx[i] = int32(i)
	}
	start := time.Now()
	for r := 0; r < rounds; r++ {
		src, err := tensor.FromSlice(vals, tensor.Shape{n}, b)
		if err != nil {
			fmt.Fprintf(os.Stderr, "FromSlice vals round %d: %v\n", r, err)
			os.Exit(1)
		}
		ids, err := tensor.FromSlice(idx, tensor.Shape{n}, b)
		if err != nil {
			fmt.Fprintf(os.Stderr, "FromSlice idx round %d: %v\n", r, err)
			os.Exit(1)
		}
		g := src.Gather(0, ids)
		_ = g.Data()
		if r == 0 || r == rounds-1 {
			fmt.Printf("round=%d elapsed=%s\n", r, time.Since(start))
		}
	}
	fmt.Printf("ok rounds=%d elapsed=%s\n", rounds, time.Since(start))
}
