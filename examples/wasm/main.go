//go:build js

// WASM 批预测 demo：Native CPU 后端，无 Born/WebGPU 依赖。
package main

import (
	"bytes"
	_ "embed"
	"fmt"
	"syscall/js"

	"github.com/linkerlin/leaves/io"
	"github.com/linkerlin/leaves/model"
	"github.com/linkerlin/leaves/tree"
)

//go:embed model.json
var modelJSON []byte

func main() {
	result, err := io.ParseXGBoostJSON(bytes.NewReader(modelJSON))
	if err != nil {
		panic(err)
	}
	outType, transform := io.ObjectiveToTransform(result.Objective, true)
	result.IR.NOutputGroups = io.NOutputGroupsForTransform(result.IR.NRawOutputGroups, outType)
	m, err := model.NewEnsembleFromIRWithHint(
		result.IR, transform, outType, tree.BackendNative, tree.DefaultWorkloadHint(),
	)
	if err != nil {
		panic(err)
	}

	js.Global().Set("leavesPredict", js.FuncOf(func(_ js.Value, args []js.Value) any {
		if len(args) < 1 {
			return errString("need feature array")
		}
		arr := args[0]
		n := arr.Length()
		if n == 0 {
			return errString("empty features")
		}
		fvals := make([]float64, n)
		for i := 0; i < n; i++ {
			fvals[i] = arr.Index(i).Float()
		}
		p := m.PredictSingle(fvals, 0)
		return js.ValueOf(p)
	}))
	js.Global().Set("leavesReady", js.ValueOf(true))

	fmt.Println("leaves wasm ready")
	select {}
}

func errString(msg string) js.Value {
	return js.Global().Get("Error").New(msg)
}
