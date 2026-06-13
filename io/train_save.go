package io

import "github.com/dmitryikh/leaves/model"

// SaveTrainModel 保存训练产出（leaves.json），与 LoadFromFile 对称。
func SaveTrainModel(path string, ir *model.ModelIR, objective string) error {
	return SaveLeavesJSONFile(path, ir, objective)
}
