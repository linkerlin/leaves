package train

import (
	"encoding/json"
	"fmt"
	"os"

	leavesio "github.com/linkerlin/leaves/io"
)

// Checkpoint 训练检查点。
type Checkpoint struct {
	Round     int     `json:"round"`
	Objective string  `json:"objective"`
	Model     json.RawMessage `json:"model"`
}

// SaveCheckpoint 保存当前训练状态。
func (l *Learner) SaveCheckpoint(path string, round int) error {
	if l.booster == nil {
		return fmt.Errorf("train: not fitted")
	}
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	if err := leavesio.SaveLeavesJSON(f, l.Model(), l.cfg.Objective); err != nil {
		return err
	}
	// 覆写为 checkpoint 包装：先写临时 leaves，再读回 — 简化为同文件仅 model
	_ = round
	return nil
}

// SaveCheckpointFile 保存 round + model 到 checkpoint JSON。
func SaveCheckpointFile(path string, round int, l *Learner) error {
	if l == nil || l.booster == nil {
		return fmt.Errorf("train: not fitted")
	}
	var modelBuf []byte
	{
		f, err := os.CreateTemp("", "leaves-ckpt-*.json")
		if err != nil {
			return err
		}
		if err := leavesio.SaveLeavesJSON(f, l.Model(), l.cfg.Objective); err != nil {
			f.Close()
			return err
		}
		f.Close()
		modelBuf, err = os.ReadFile(f.Name())
		os.Remove(f.Name())
		if err != nil {
			return err
		}
	}
	ckpt := Checkpoint{
		Round:     round,
		Objective: l.cfg.Objective,
		Model:     json.RawMessage(modelBuf),
	}
	out, err := os.Create(path)
	if err != nil {
		return err
	}
	defer out.Close()
	enc := json.NewEncoder(out)
	enc.SetIndent("", "  ")
	return enc.Encode(ckpt)
}

// LoadCheckpointFile 加载检查点中的 model（leaves.json 子文档）。
func LoadCheckpointFile(path string) (round int, objective string, modelJSON []byte, err error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return 0, "", nil, err
	}
	var ckpt Checkpoint
	if err := json.Unmarshal(b, &ckpt); err != nil {
		return 0, "", nil, err
	}
	return ckpt.Round, ckpt.Objective, ckpt.Model, nil
}

// BestRound 返回早停最优轮次（未启用早停时返回 0）。
func (l *Learner) BestRound() int {
	if l.cfg.EarlyStop == nil {
		return 0
	}
	return l.cfg.EarlyStop.BestRound()
}
