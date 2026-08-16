# leaves v2.5.2

**主题**：`--from-run` 谱系流程修复——tag 未命中不再硬错。

## Highlights

1. **`train --from-run --tag <新tag>` 回落最优行**：tag 不在账本时原报 `usage` 错误退出，导致 SKILL §4.5 演化谱系流程（「从最优复现 + `--tag p:parent+mutation` 给子代起新名」）无法直接组合。现在：回落**最优行**作父代，stderr 注明 `已回落最优行 tag=<父代>`（笔误可审计发现），用户新 tag 原样写入本次账本行。

## Usage

```powershell
# 谱系流：从账本最优复现，起谱系新名
leaves train --data train.csv --from-run runs.jsonl --tag p:best+depth8 `
  --val hold.csv --early-stop 15 --cv 0 --out-model m2.leaves.json --metrics m2.json --runs runs.jsonl
# stderr: 注意：runs.jsonl 中无 tag="p:best+depth8"，已回落最优行 tag="..."
```

注意：父代行含 `cv_folds` 时会连带 CV 路径；切 `--val --early-stop` 单跑路径须显式 `--cv 0`（SKILL 已注明）。

## Verification

- `TestFromRunReproduce` 扩展：新 tag → 回落 best 行 params（depth/lr 断言）+ 新 tag 原样落账本（第 4 行）。
- 仓库外用户路径实测：`go install` 二进制 + 临时数据 `sniff → CV 基线 → --from-run 谱系回落 → publish` 全通。
- `go test ./cmd/leaves ./docs -count=1` 绿。

## Compatibility

- **Breaking**：无。原先的硬错误场景（非零退出）变为回落 + stderr 提示；脚本若依赖该错误行为（不合理）需改用 `--strict-flags` 类校验。
