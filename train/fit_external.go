package train

import (
	"fmt"

	"github.com/linkerlin/leaves/data"
	"github.com/linkerlin/leaves/treebuilder"
)

// resolveTrainMatrix 外存多批次 + exact 时物化；hist 保留 BatchedMatrix。
func (l *Learner) resolveTrainMatrix(dm data.Matrix) (data.Matrix, error) {
	em := data.AsExternalMemoryMatrix(dm)
	if em == nil || em.NumBatches() <= 1 {
		return dm, nil
	}
	requested := l.cfg.TreeMethod
	if requested == "" {
		requested = treebuilder.MethodAuto
	}
	resolved, _ := ResolveTrainTreeMethod(requested, dm.NumRow())
	if resolved == treebuilder.MethodExact {
		dense, err := data.MaterializeExternal(em)
		if err != nil {
			return nil, fmt.Errorf("train: materialize external: %w", err)
		}
		return dense, nil
	}
	return dm, nil
}

// FitExternal 显式外存训练入口。
func FitExternal(cfg Config, em data.ExternalMemoryMatrix) (*Learner, error) {
	learner, err := NewLearner(cfg)
	if err != nil {
		return nil, err
	}
	if err := learner.Fit(em); err != nil {
		return nil, err
	}
	return learner, nil
}
