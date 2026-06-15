package train

import (
	"fmt"
	stdio "io"
	"os"

	leavesio "github.com/linkerlin/leaves/io"
	"github.com/linkerlin/leaves/model"
)

func saveLeavesJSON(path string, ir *model.ModelIR, objective string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	if err := leavesio.SaveLeavesJSON(f, ir, objective); err != nil {
		return err
	}
	return nil
}

// Save 保存训练模型为 leaves.json。
func (l *Learner) Save(path string) error {
	if l.booster == nil {
		return fmt.Errorf("train: not fitted")
	}
	return saveLeavesJSON(path, l.Model(), l.cfg.Objective)
}

// SaveTo 写入 leaves.json 到任意 Writer。
func (l *Learner) SaveTo(w stdio.Writer) error {
	if l.booster == nil {
		return fmt.Errorf("train: not fitted")
	}
	return leavesio.SaveLeavesJSON(w, l.Model(), l.cfg.Objective)
}
