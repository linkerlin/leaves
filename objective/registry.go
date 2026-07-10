package objective

import "fmt"

// factory 按名称构造目标；numClass 仅 multi:* 使用，其它可忽略。
type factory func(numClass int) (Func, error)

var extra = map[string]factory{}

// Register 注册自定义或覆盖内置目标。
// 内置目标在 builtins.go init() 中注册（含 multi:* / rank:*）。
// 训练侧排序超参仍经 ConfigureRanking 覆盖默认 RankTrainConfig。
func Register(name string, f factory) {
	extra[name] = f
}

// RegisteredNames 返回当前已注册目标名（无序；含空串别名时出现 ""）。
func RegisteredNames() []string {
	out := make([]string, 0, len(extra))
	for k := range extra {
		out = append(out, k)
	}
	return out
}

// ByNameWithClass 解析目标函数（多分类需 numClass >= 2）。
// 排序目标使用默认 RankOptions；训练侧通过 ConfigureRanking 覆盖。
func ByNameWithClass(name string, numClass int) (Func, error) {
	if f, ok := extra[name]; ok {
		return f(numClass)
	}
	return nil, fmt.Errorf("objective: unsupported %q (register with objective.Register)", name)
}

// ConfigureRanking 用训练超参覆盖排序目标选项。
func ConfigureRanking(obj Func, cfg RankTrainConfig) Func {
	switch obj.(type) {
	case RankPairwise:
		return NewRankPairwise(cfg)
	case RankNDCG:
		return NewRankNDCG(cfg)
	case RankListwise:
		return NewRankListwise(cfg)
	default:
		return obj
	}
}

// IsMulticlass 判断是否为多分类目标。
func IsMulticlass(obj Func) (*Multiclass, bool) {
	m, ok := obj.(Multiclass)
	if !ok {
		return nil, false
	}
	return &m, true
}
