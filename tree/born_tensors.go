package tree

import (
	"fmt"

	"github.com/born-ml/born/tensor"
)

func bornMustFromF64[B tensor.Backend](b B, data []float64, shape tensor.Shape) *tensor.Tensor[float64, B] {
	t, err := tensor.FromSlice(data, shape, b)
	if err != nil {
		panic(fmt.Sprintf("born: FromSlice f64: %v", err))
	}
	return t
}

func bornMustFromI32[B tensor.Backend](b B, data []int32, shape tensor.Shape) *tensor.Tensor[int32, B] {
	t, err := tensor.FromSlice(data, shape, b)
	if err != nil {
		panic(fmt.Sprintf("born: FromSlice i32: %v", err))
	}
	return t
}

func bornDenseFeatures[B tensor.Backend](b B, vals []float64, rows, cols int) *tensor.Tensor[float64, B] {
	t, err := tensor.FromSlice(vals, tensor.Shape{rows, cols}, b)
	if err != nil {
		panic(err)
	}
	return t
}

func bornGatherFeatureValues[B tensor.Backend](
	features *tensor.Tensor[float64, B],
	featIdx *tensor.Tensor[int32, B],
) *tensor.Tensor[float64, B] {
	return features.Gather(1, featIdx.Unsqueeze(1))
}
