---
name: leaves-autotrain
description: >-
  指导 Agent 用 leaves 通用 CLI 完成「训练 → 指标评估 → 超参优化 → 发布」全自动闭环。
  不用 MCP：Agent 通过 shell 调 `leaves` 命令、读写 metrics.json 驱动循环；Agent 本身即优化器。
  凡涉自动训练、调参、指标优化、模型发布、AutoML、leaves CLI 时用之。
---

# leaves 自动训练 / 优化 / 发布

> Agent 是优化器，leaves 是黑盒目标函数；CLI 出模型，metrics.json 是唯一闭环信号。

## 一、何时激活

- 用户说「训练模型」「自动调参」「优化指标」「把指标提上去」「发布模型」「AutoML」
- 需要在任意数据上跑回归 / 分类 / 排序，而非推荐系统专用流水线
- 已有 MCP 但本路径**故意不用**：shell + JSON 更可复现、可审计

> 推荐系统（召回→排序→发牌）走 [`recsys-orchestrator`](../recsys-orchestrator/SKILL.md)；
> 本 SKILL 面向**任意监督学习任务**的单模型训练与发布。

## 二、核心闭环（Agent 驱动）

```text
  ┌───────────────────────────────────────────────────────┐
  │  0. 建账本  runs.jsonl（Agent 跨迭代记忆，见 §二.5）     │
  │  1. 嗅探数据  leaves sniff --data X → suggested_objective │
  │              （自动识别二分类/多分类/回归/排序，无需人工告知）│
  │  2. 基线训练  → metrics.json + 追加 runs.jsonl           │
  │  3. 读 metrics.json 的 value / cv_mean                  │
  │  4. 按决策表（§四）选下一组超参 ← Agent 的"大脑"在这里     │
  │  5. 再训（可 --cv K 取更稳估计）                          │
  │  6. 收敛判据（§五）满足？否 → 回 4                        │
  │  7. 是 → leaves inspect 复核 → leaves publish 出工件包   │
  └───────────────────────────────────────────────────────┘
```

**Agent 不写 Go 代码**。每一步都是一条 shell 命令；判断完全基于 metrics.json 的数值与决策表。
**工作流由 Agent 动态构建**：leaves 不内置固定流水线，Agent 按 SKILL 建议按需串联命令。

## 二、5 运行账本（runs.jsonl）—— 动态优化的记忆

没有记忆的优化是盲搜。每次 `train --runs runs.jsonl --tag <名>` 会在账本追加一行：

```json
{"tag":"baseline","ts":"2026-07-09T04:06:20Z","metric":"rmse","value":0.236,"maximize":false,"cv_mean":0.241,"cv_std":0.07,"params":{"rounds":50,"depth":6,"lr":0.3,...}}
```

Agent 每轮决策前 `Get-Content runs.jsonl`（或读全文件）：

- **选最优**：按 `maximize` 在所有记录里取 `value` 最优的一组 `params`，作为下一轮起点与发布候选。
- **一键复现**：`leaves train --data X --from-run runs.jsonl --tag <最优tag> --out-model ...`（无 `--tag` 则自动取最优行；其余 CLI flag 覆盖账本）。
- **避免重复**：若新提议的 `params` 已在账本中且更差，跳过。
- **可恢复**：长会话中断后，Agent 从账本续起，无需重头搜。
- **收敛判据**（§五）的「连续 3 次改进 <0.5%」直接读账本尾部判定。

> 账本由 CLI 可靠追加（不依赖 Agent 手写 JSON）；Agent 只读不写账本。

## 三、CLI 速览（详见 [cli.md](cli.md)）

```powershell
# 0) 嗅探数据 → 自动推荐 objective/metric（不必先问用户任务类型）
leaves sniff --data train.csv --metrics sniff.json

# 基线（带 5 折 CV，输出 metrics.json + 模型，并记入账本）
leaves train --data train.csv --objective reg:squarederror `
  --eval-metric rmse --cv 5 --rounds 50 --depth 6 --lr 0.3 `
  --out-model m.leaves.json --metrics metrics.json `
  --runs runs.jsonl --tag baseline

# 复核模型状态（objective/特征/树数，发布前用）
leaves inspect --model m.leaves.json --metrics inspect.json

# 诊断：特征重要性（Agent 查"模型为什么不改进"）
leaves explain --model m.leaves.json --type importance --metrics imp.json

# 对比已存模型
leaves eval --model m.leaves.json --data holdout.csv --eval-metric rmse `
  --metrics eval.json

# 生成预测（部署验证用）
leaves predict --model m.leaves.json --data holdout.csv --out pred.jsonl

# 发布本地产物包（leaves.json + 可选量化/XGB 导出 + manifest）
leaves publish --model m.leaves.json --out-dir release/v1 `
  --version 1.0.0 --quantize --export-xgb --metrics metrics.json
```

> 若仓库内尚未 `go install`，用 `go run ./cmd/leaves ...` 等价。

## 四、决策表（Agent 的调参大脑）

### 4.1 任务 → objective / metric

| 任务 | objective | 推荐 eval-metric | maximize |
|------|-----------|------------------|----------|
| 回归 | `reg:squarederror` | `rmse`（或 `mae`/`mape`） | 否 |
| 二分类 | `binary:logistic` | `logloss`（或 `auc`） | logloss 否 / auc 是 |
| 多分类 | `multi:softmax`（+`--num-class`） | `mlogloss` | 否 |
| 计数 | `count:poisson` | `rmse` | 否 |
| 零膨胀/保险 | `reg:tweedie` | `rmse` | 否 |
| 生存 | `survival:cox` | `cox-nloglik`（无则 `rmse`） | 否 |
| 排序 | `rank:ndcg` | `ndcg@10` | 是 |

> 排序数据须为 `qid label feat...` 的 ranking TSV（同 qid 行相邻）；非排序用 CSV/LIBSVM。
> `data.FromFileAuto` 自动嗅探，无需手动指定格式。

### 4.2 超参旋钮 → 效果方向

| 旋钮 | flag | 调大时的倾向 | 何时去调 |
|------|------|--------------|----------|
| 轮数 | `--rounds` | train↓；val 先↓后↑（过拟合拐点） | 配早停找拐点 |
| 深度 | `--depth` | 拟合更强、更易过拟合 | 欠拟合时↑；过拟合时↓ |
| 叶数 | `--max-leaves` | lossguide 下替代 depth | 想用叶数控模型大小时设>0 |
| 学习率 | `--lr` | 越小越稳越慢 | `--lr`↓ 同时 `--rounds`↑ 通常更优 |
| L2 | `--lambda` | 抑制过拟合 | 过拟合时↑ |
| min_child_weight | `--min-child-weight` | 更保守 | 过拟合时↑ |
| gamma | `--gamma` | 分裂更少、更保守 | 过拟合时↑ |
| subsample | `--subsample` | 行采样抗过拟合（排序禁用） | 过拟合时降到 0.6~0.9 |
| colsample | `--colsample` | 列采样抗过拟合 | 过拟合时降到 0.5~0.9 |
| max_bin | `--max-bin` | 越粗越快越抗噪 | 训练慢/噪声大时↓ |

**经验搜索顺序**（省 token）：先定 `--depth` 与 `--lr/--rounds` 组合 → 再调 `--lambda/--min-child-weight` → 最后 `--subsample/--colsample`。每轮只动 1~2 个旋钮。

### 4.3 过拟合 vs 欠拟合判据（读 metrics.json）

- `train_metric` 远好于 val/CV → 过拟合：调小 `--depth`、调大 `--lambda/--min-child-weight/--gamma`、加 `--subsample/--colsample`、减 `--rounds` 或开早停。
- train 与 val 都差 → 欠拟合：调大 `--depth`、调大 `--rounds`、调大 `--lr`、扩特征。

### 4.4 搜索策略（让"动态构建"有章法）

**开局三步**（任何任务都从这开始，不跳）：
1. `leaves sniff --data X` → 取 `suggested_objective`/`suggested_metric`，直接用。**不必问用户任务类型。**
   - 若为 `multi:softmax`，取 sniff 输出的 `label.n_unique` 作为 `--num-class`。
2. 默认超参（depth=6, lr=0.3, rounds=50）跑一次 `--cv 5` 基线 → 定基准线。
3. 读 `train_metric` 与 `cv_mean` 的差，决定主攻方向（§4.3）。

**调参网格**（按顺序，每轮只动 1–2 个旋钮；用账本去重）：
- 三角：`(depth=4, lr=0.1, rounds=200)` 与 `(depth=8, lr=0.05, rounds=400)` 各试一次，挑更优方向。
- 正则：`lambda ∈ {3, 10}`、`min_child_weight ∈ {5, 10}`、`gamma ∈ {1, 5}`。
- 采样（非排序）：`subsample=0.8, colsample=0.8`。
- 计数/偏斜：试 `max_bin ∈ {64, 128}` 抗噪。

**卡住恢复表**（账本尾部无明显改进时查这张表，而不是盲目加轮）：

| 症状 | 对策 |
|------|------|
| `cv_std` 大（切分敏感） | 停止单调加轮；加正则或扩数据；换更浅树+更强正则 |
| 指标 plateau 不动 | 换 `tree-method`（hist↔exact）；调 `seed` 看是否噪声；**跑 `explain --type importance` 看特征权重**——若某特征 gain≈0 可能是噪声 |
| train 好但 holdout 崩 | 数据漂移/切分泄漏；大砍 `depth`/`rounds`，开 `--early-stop` |
| 二分类 AUC≈0.5 | 先 `sniff` 看 `label.n_unique` 与均衡度；标签或特征基本无关 |
| 排序 NDCG 不升 | 查 `--ndcg-k` 是否与业务一致；确认 group 完整（同 qid 相邻） |
| subsample 反而变差 | 数据太小（<1000 行）→ subsample 砍样本得不偿失，去掉 |

**CV vs holdout 判据**：

- 行数 <10k：用 `--cv 5`（数据少，不宜再切 holdout）。
- 行数 ≥50k：切一份 holdout；优化期用 `--cv 3`，**定稿前**用 `eval` 在 holdout 上独立验收。
- 排序任务：CV 可能切散 group，优先用 `--val`（保持 group 完整）。

> 经验：80% 的收益来自前三步（sniff→基线→一次三角调整）。后续正则/采样是边际优化，达到 §五 收敛判据即停，不要无限调。

---

## 五、收敛与停止判据

满足任一即可停止优化、进入发布：

- CV `value` 连续 **3 次** 调参改进 < **0.5%**（收益低于噪声）。
- 已遍历 §4.2 经验顺序的关键组合。
- `train` 与 `val/CV` 差距开始单调放大（过拟合主导），取此前的最佳 metrics.json。
- 指标已达用户目标值。

> **定稿语义（默认已安全）**：`--early-stop` 时默认 `--save-best`（可省略），磁盘模型截断为 **best_round**，metrics 中 `model_round == best_round`。仅当需要 final-round 轨迹研究时显式 `--save-best=false`。旧版「必须按 best_round 重训」流程已不需要。

> 用 `--cv 5` 时以 `cv_mean` 为主、`cv_std` 看稳定性；`cv_std` 大说明对切分敏感，优先加正则而非堆轮数。

## 六、发布检查表（`leaves publish`）

`--out-dir` 下应出现：

- [x] `model.leaves.json`（主模型，可 `io.LoadFromFile` 推理）
- [x]（`--quantize`）`model.quant.json`（int8 量化侧车，`predict` 可直接加载重建量化推理）+ `quantize_report.json`（parity + max_threshold_err）
- [x]（可选）`model.xgb.json`（`--export-xgb`，leaves→XGBoost 3.x）
- [x] `manifest.json`：`version`、`objective`、`num_features`、`n_trees`、各文件 `sha256`、`metrics` 快照、`created_at`

**发布=本地工件包**。leaves 明确不做 serving/registry（见 TODO.md「明确不做」）；推送到镜像/registry 由用户的 CI 在库外接。

## 七、完成标准

- [ ] 最佳 metrics.json 的 `value` 可复现（同 `--seed` 重训一致）
- [ ] `leaves eval` 在独立 holdout 上的指标与训练时一致（±数值噪声）
- [ ] `leaves publish` 产物齐全且 manifest 各 sha256 校验通过
- [ ] 模型可被 `io.LoadFromFile` 加载并 `PredictDense`

## 八、边界（不覆盖）

- 不内置网格/贝义斯搜索（搜索逻辑在 Agent 文本里，不在 leaves 代码里——设计如此）
- 不做分布式训练 / 实时在线学习 / serving 框架 / registry 推送
- 召回算法 / 发牌策略属推荐系统，见 `recsys-*` SKILL
- **数据要求**：数值型 CSV（表头可选，label 列名含 label/target/y/class 时自动识别）/ LIBSVM / 排序 TSV（qid label feat…）。默认 `--na-policy error`（空/NA 单元格失败）；可选 `skip-row` 丢弃含缺失的整行（**不做插补**）。非数值列须预先编码。`sniff` 的 `data_quality.warnings` 可预知常数列/不均衡/小样本。

### 何时用 autotrain vs recsys

| 场景 | 用哪套 |
|------|--------|
| 任意监督表 / 单模型调参发布 | **本 SKILL**（leaves-autotrain） |
| 召回→LTR→发牌全链路 | [`recsys-orchestrator`](../recsys-orchestrator/SKILL.md) |
| 仅精排（数据已是 ranking TSV） | autotrain 或 [`recsys-rank`](../recsys-rank/SKILL.md) |

CLI flag 全表与 metrics.json schema 见 [`cli.md`](cli.md)。零代码 demo 见 [`examples/autotrain/README.md`](../../examples/autotrain/README.md)。

---

## 附录：Agent 推理 walkthrough（`examples/autotrain/` 实战，数字已验证 2026-07-10）

> Agent 拿到 `examples/autotrain/data/train.csv`（120 行、3 特征、回归，`y≈2x₀−1.5x₁+0.8x₂+微噪`）与 `holdout.csv`（30 行），按下面的推演完成全闭环。**以下数字均为实跑结果（seed=42），可复现。**
>
> **路径纪律**：`--out-model` 与 `--metrics` **必须不同路径**（否则 metrics 覆盖模型，inspect 失败）。推荐 `mN.leaves.json` + `mN.metrics.json`。

### 第 0 轮：识别任务

```
> go run ./cmd/leaves sniff --data examples/autotrain/data/train.csv --metrics sniff.json
→ {"format":"csv","n_rows":120,"n_features":3,"feature_names":["x0","x1","x2"],
   "label":{"kind":"regression"},"suggested_objective":"reg:squarederror","suggested_metric":"rmse",
   "data_quality":{"numeric":true,"nan_cells":0,"warnings":[]}}
```
**Agent 推理**：3 特征、连续标签 → `reg:squarederror` + `rmse`。无需问用户。

### 第 1 轮：基线（CV 5 折——诚实估计）

```
> go run ./cmd/leaves train --data .../train.csv --objective reg:squarederror
  --cv 5 --rounds 40 --depth 4 --lr 0.2
  --out-model m1.leaves.json --metrics m1.metrics.json --runs runs.jsonl --tag baseline
```
`m1.metrics.json`：
```json
{"metric":"rmse","value":0.2314,"maximize":false,"cv_mean":0.2314,"cv_std":0.0089}
```
**Agent 推理**：CV 基线 0.2314，cv_std=0.0089（约 3.8% of mean — 切分稳定）。此轮用 `--cv` 获取诚实估计；CV 路径不提供逐轮诊断。下一轮：切到 `--val` + `--early-stop` + `--emit-rounds` 做详细调参。

### 第 2 轮：加深 + 降学习率 + 早停（关键！）

```
> go run ./cmd/leaves train --data .../train.csv --objective reg:squarederror
  --rounds 200 --depth 6 --lr 0.1
  --val .../holdout.csv --early-stop 20 --emit-rounds rounds.jsonl
  --out-model m2.leaves.json --metrics m2.metrics.json --runs runs.jsonl --tag depth6_lr01
```
`m2.metrics.json`：
```json
{"value":0.218,"train_metric":0.139,"best_round":43,"stopped_round":63,"model_round":43}
```
**Agent 推理**：best val=0.218（早停报告 best-round），比基线 0.2314 降 5.8% ✓。`best_round=43` / `model_round=43`（默认 save-best 已截断）。`train_metric=0.139` 是 **截断后** 模型在训练集上的指标（非 final 200 轮）。**这是目前最优。** 下一轮：试加正则看能否超越 0.218。

### 第 3 轮：强正则（未加早停——反面教材）

```
> go run ./cmd/leaves train ... --rounds 200 --depth 6 --lr 0.1 --lambda 5 --min-child-weight 10
  --val .../holdout.csv
  --out-model m3.leaves.json --metrics m3.metrics.json --runs runs.jsonl --tag reg_strong
```
`m3.metrics.json`：
```json
{"value":0.2229,"train_metric":0.1293,"model_round":200}
```
**Agent 推理**：val=0.2229 > R2 的 0.218 → 退化！**但注意**：R3 未加 `--early-stop`，val 是 final-round（第 200 轮）的值——已过拟合。强正则（lambda=5）没带来收益。读 `runs.jsonl`：最优仍是 `depth6_lr01`(0.218)。

### 第 4 轮：温和正则 + 采样（小数据上的反面教材）

```
> go run ./cmd/leaves train ... --rounds 200 --depth 6 --lr 0.1 --lambda 2 --min-child-weight 3
  --subsample 0.8 --colsample 0.8
  --val .../holdout.csv
  --out-model m4.leaves.json --metrics m4.metrics.json --runs runs.jsonl --tag reg_mild
```
`m4.metrics.json`：
```json
{"value":0.2309,"train_metric":0.129}
```
**Agent 推理**：val=0.2309 → 最差！`subsample=0.8` 在仅 120 行上砍掉 20% 样本。**教训：小数据（<1000 行）不要用 subsample/colsample。** 账本：baseline(0.2314)→depth6_lr01(0.218)→reg_strong(0.2229)→reg_mild(0.2309)。

### 收敛判定与发布

- 最优 run = `depth6_lr01`(value=0.218, best_round=43, model_round=43)。
- 后续两轮均退化 → 收敛。
- **定稿**：默认 save-best 已使磁盘模型 = best_round，**无需手写重训**。也可用一键复现：
  `train --data ... --from-run runs.jsonl --tag depth6_lr01 --out-model final.leaves.json ...`
  （params 从账本加载，CLI flag 可覆盖）。

```
> go run ./cmd/leaves inspect --model m2.leaves.json
  → {"objective":"reg:squarederror","n_trees":43,"num_features":3,"kind":"gbtree"} ✓
> go run ./cmd/leaves publish --model m2.leaves.json --out-dir release/v1 --version 1.0.0
  --quantize --data .../train.csv --export-xgb --metrics m2.metrics.json
```

产物：`model.leaves.json` + `model.quant.json`（量化侧车，parity pass✓）+ `model.xgb.json` + `manifest.json`（含 `reproduce`）。

**总结**：全程读 JSON/JSONL、零 Go 代码；(1) 早停关键；(2) 小数据别 subsample；(3) save-best 保证发布模型是最优轮；(4) `--out-model`≠`--metrics`；(5) `--from-run` 降低复现拼 flag 错误。

---

## 速查卡（Agent 缓存用）

```
⸻ 8 命令速查 ⸻

sniff    --data FILE                    → suggested_objective / feature_names / label stats
train    --data FILE --objective OBJ    → metrics.json + --out-model + --emit-rounds + --runs/--tag
             --cv K  --val FILE  --early-stop N
             --from-run runs.jsonl [--tag NAME]  # 账本复现；CLI 覆盖优先
eval     --model FILE --data FILE       → metrics.json（对比已存模型）
predict  --model FILE --data FILE       → JSONL 或 --format csv（部署）
inspect  --model FILE                   → {objective, kind, n_trees, num_features, n_output_groups}
explain  --model FILE [--type importance] [--data FILE]  → 特征重要性/SHAP
publish  --model FILE --out-dir DIR     → 本地工件包 + manifest（--quantize --export-xgb --emit-repro-script）

⸻ 最常见 flags ⸻

训练：--rounds --depth --lr --lambda --min-child-weight --subsample --colsample --seed
早停：--early-stop N（配 --val）；默认 --save-best（model_round==best_round）
诊断：--emit-rounds rounds.jsonl --runs runs.jsonl --tag NAME
发布：--quantize --export-xgb --emit-repro-script both --version
缺失：--na-policy error|skip-row（默认 error；不做插补）

⸻ 优化铁律 ⸻

1. 先 sniff → 自动定 objective
2. 默认超参 baseline → 看 train/cv 差距 → 定过拟合/欠拟合方向
3. 每轮只动 1-2 个旋钮 → runs.jsonl 去重 → 收敛即停
4. plateau → run explain --type importance 看特征权重
5. 收敛 → inspect 复核 → publish 定稿
```
