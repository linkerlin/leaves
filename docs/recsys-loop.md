# 推荐生产闭环控制面（RC）

> **文档类型**：RC-00 现状基线 + RC 控制面使用指南  
> **依据**：[`演进方案.md`](../演进方案.md) §十七（RC0–RC4 工作包）  
> **日期**：2026-08-21 · **状态**：RC0–RC3 契约与测试已落地；RC-09 只含 fake adapter

---

## 1. 现状基线（RC-00：离线 deal ≠ 线上 decision）

`recsys` 四段流水线（prep → recall → rankconv/trainrank → deal）产出的是**离线
deal 终稿文件**：`deal/deal_test.tsv`。它是一次可重复的离线推荐结果，**不是**
一次线上推荐决策——没有事件时间、曝光事实、反馈归因、策略版本与上线状态。

生产闭环需要的是：

```text
可验证数据快照 → 离线候选/排序/发牌 → 候选工件
      ↑                              ↓
  反馈归因 ← 决策/曝光账本 ← 外部 serving / registry / rollout
      ↓                              ↓
 监控与重放 ───── Agent 再训练与推广门禁 ──→ 回滚或提升
```

| 已有（离线） | 缺口（生产，本文档覆盖的控制面契约） |
|--------------|--------------------------------------|
| 四段流水线 + NDCG 评估 | 事件/快照/决策/曝光/反馈/证据契约（§2） |
| 用户切分（prep） | 时间切分 + as-of 因果（§3） |
| 单一 NDCG | 三层离线门禁（§4） |
| deal 终稿 + 日志 | 决策/曝光/反馈账本（§5） |
| — | 归因窗回放与再训练样本（§6） |
| — | 窗口监控 ok/warn/block（§7） |
| publish 本地工件包 | 发布状态机 + evidence + adapter 请求（§8） |

## 2. 契约（`recsys/contract`）

冻结 schema v1；全部事件带 UTC 时间；ID 为匿名稳定键（拒空白/`@` 绊线）；
字段只增不删。类型与校验见 `recsys/contract/contract.go`：

| 契约 | 用途 |
|------|------|
| `DatasetSnapshot` | 不可变数据快照：输入文件 sha256 + `FeatureSchemaHash` + 时间范围；`VerifyFiles` 任一 hash 不匹配即失败 |
| `InteractionEvent` | impression/click/conversion/rating 事件 |
| `DecisionEvent` | 一次排序+发牌决策（**审计真源**；Top-K 带原因码 ok/tag_overflow/fallback/…） |
| `ExposureEvent` | 曝光；必须回链决策（时间、条目、位次一致） |
| `FeedbackEvent` | 反馈；必须回链曝光（或仅 decision——不构成 supervised 正样本） |
| `ReleaseEvidence` | 发布证据：模型 hash、快照、离线指标、三层门禁、审批、回滚目标 |

旧四元数据经 `LegacySnapshot=true` 显式导入，**不得**进入时间因果门禁。

## 3. 时间切分（`recsys/split`）

`split.Split` 按 `train_end < validation_start < test_start` 切分；
`split.CheckLeakage` 断言训练事件全部早于 eval 起点；隔离带事件丢弃。
用户切分仅保留为 cold-start 附加实验（`ColdStartUsers`）。

## 4. 离线门禁（`recsys/eval`）

三层指标（`evaluation.json`）：数据（`data_leakage_rate` 等）、候选排序
（`recall_at_k`/`ndcg_at_k`/`map_at_k`/`coverage`）、发牌（`deck_fill_rate`/
`deck_dup_rate`/`deck_tag_overflow_rate`/`deck_item_coverage`）。
阈值是**业务配置**；阈值缺失 → `purpose=exploratory`，不得进入 candidate；
未知指标 → `block`（integrity）。

## 5. 决策账本（`recsys/ledger`）

`ledger.jsonl` 追加写（decision/exposure/feedback 信封行）。append 时校验：
曝光回链决策（含位次一致）、反馈回链曝光、时间不倒流、ID 不重复。
`ledger.DecisionFromDeal` 把 deal 终稿 + LogEntry 映射为决策事件
（补位条目带 `tag_overflow` 原因码）。

## 6. 回放（`recsys/replay`）

`replay.BuildSamples`：只有可归因到 shown 曝光的窗内反馈构成正样本；
迟到/孤立/抑制曝光反馈计数入 `replay_report.json` 的 `diff_reasons`，
不进样本。确定性：同输入同输出（按曝光时间 + ID 稳定排序）。

## 7. 监控与触发器（`recsys/monitor`）

`monitor.BuildReport` 聚合窗口内账本指标（ctr/orphan_feedback_rate/曝光量）
+ 发牌质量 → `monitor_report.json`，状态 `ok|warn|block` + reason code；
无阈值 → `unevaluated`。指标/完整性硬失败 → block。

**触发器**（§17.6：带窗口与冷却期的可配置规则，不是 Agent 临场猜测）：
`monitor.NewTriggerSet` 定义规则——指标连续 N 个窗口越界（warn/block 级别）
→ `retrain_requested` 或 `rollback_requested`；`Cooldown` 抑制重复触发；
恢复自动重置连续计数。安全类规则用 `Level=block + Consecutive=1`（立即）。
Agent/编排层拿 `Evaluate` 返回的 `Fired` 去调 `release.Machine` 的
`RequestRetrain/RequestRollback`——动作仍走状态机，触发器只做决策。

## 8. 受控发布（`recsys/release`）

状态机：`exploratory → candidate → approved → promoted → observing →
(retrain_requested | rollback_requested | retired)`。

- `ToCandidate`：evidence 完整 + 三层门禁齐全无 block + 模型文件 hash 与记录一致；
- `Approve`：人工批准默认必经；
- `Observe`：promoted 版本成为 `last_known_good` 锚点（或显式 `SetLastKnownGood`，
  必须指向已记录证据，不可隐式漂移、不可指向自身）；
- `RequestRollback`：只指向锚点；产出 adapter-neutral 请求；
- `release.Adapter` 接口 + `FakeAdapter`：真实 HTTP/registry/CI adapter 留给应用仓库，
  leaves 不执行网络副作用；
- 每次变迁写 `run_status.jsonl`（Agent 只依赖结构化状态 + 退出码）。

## 9. 端到端演练（`recsys/loop`，RC-11）

`TestAgenticRecsysLoopDrill` 以合成确定性事件驱动 §1 图的完整闭环：
快照/切分 → 门禁 evidence → fake adapter promoted → 决策/曝光/反馈账本 →
健康窗口 observing → 注入 deck 退化 block → **触发器**产出回滚请求（含冷却抑制）
→ rollback 指向 last_known_good → replay 只消费可归因反馈 → retrain_requested + 下一轮快照。
同剧本的 shell 版见 §10（`recsys/cmd/control`）。

```powershell
go test ./recsys/... -count=1
```

## 10. 控制面 CLI（`recsys/cmd/control`）

八段剧本的 shell 入口（Agent 零 Go 代码编排）。退出码 0/1/2（1=用法/IO，2=校验/内部）；
全部输出为结构化文件；promote/rollback 请求打印到 stdout 由调用方转发给应用侧 adapter。

```text
control snapshot   -workspace DIR -out snapshot.json -snapshot-id ID -purpose train|eval|release
control split      -events events.jsonl -train-end T -val-start T -test-start T -out-dir DIR
control eval       -workspace DIR -thresholds th.json [-out evaluation.json] [-recall-k 100] ...
control from-deal  -workspace DIR -ledger ledger.jsonl -model-version V -policy-version P -occurred-at T
control append-exposure -ledger L -in exposures.jsonl
control append-feedback -ledger L -in feedback.jsonl
control replay     -ledger L -out samples.jsonl [-window 24h] [-negative impressed_no_feed|none]
control monitor    -ledger L -workspace DIR -window-start T -window-end T [-thresholds th.json] [-triggers tr.json]
control release    -state release_state.json -action candidate|approve|confirm-promote|observe|retrain|rollback|retire|status
```

要点：

- `snapshot` 自动对工作区输入文件取 sha256 + 特征指纹（`items.tsv` 推导）；
- `eval`/`monitor` 的阈值与触发器是 JSON 文件（`eval.Threshold` / `monitor.Trigger` 数组）；
- `monitor -triggers` 把 `TriggerSet.Evaluate` 结果追加到 `fired.jsonl`，Agent 据此调 `release`；
- `release` 状态机跨命令持久化于 `release_state.json`（含 evidence + history）；
  `confirm-promote`/`rollback` 打印 desired-state 请求 JSON；`candidate` 从
  `evaluation.json` + 模型文件自动组装 evidence（模型 hash 与文件强制一致）；
- 人工批准默认（`approve -approver` 必填）。

端到端演练：`go test ./recsys/cmd/control -count=1`（TestControlCLIEndToEnd 跑完八段）。

### 10.1 实跑演练（2026-08-21 实测数字，种子 42 合成数据）

```powershell
# 工作区（四段流水线产物：18 训练用户 / 6 测试用户 / 60 发牌行）
go run ./recsys/cmd/smoke -workspace rcws

# 1+2. 快照 + 时间切分（snapshot 必填 -time-start/-time-end：契约要求）
control snapshot -workspace rcws -out snapshot.json -snapshot-id snap-demo -purpose release `
  -created-at 2026-08-20T12:00:00Z -time-start 2026-08-01T00:00:00Z -time-end 2026-08-20T00:00:00Z
#   → 「快照 snap-demo 已写（3 输入文件指纹）」
control split -events events.jsonl -train-end 2026-08-06T00:00:00Z -val-start 2026-08-06T01:00:00Z `
  -test-start 2026-08-13T00:00:00Z -out-dir split/
#   → 「train=3 val=2 test=3 gap=0（泄漏检查通过）」（split_report.json: leakage_ok=true）

# 3. 三层门禁 → evaluation.json
control eval -workspace rcws -thresholds thresholds.json -out evaluation.json -recall-k 100 -event-count 200
#   → 「purpose=gate status=ok（9 指标, 4 门禁）」

# 4. 决策账本 + 事件摄取
control from-deal -workspace rcws -ledger ledger.jsonl -model-version m-demo -policy-version deal-v1 -occurred-at 2026-08-20T12:00:00Z
#   → 「追加 6 条决策（6 用户）」
control append-exposure -ledger ledger.jsonl -in exposures.jsonl   # 「追加 2 条曝光」
control append-feedback -ledger ledger.jsonl -in feedback.jsonl    # 「追加 1 条反馈」

# 5. 归因回放
control replay -ledger ledger.jsonl -out samples.jsonl -report replay_report.json -window 24h
#   → 「回放: 正=1 负=1（迟到=0 孤立=0）」

# 6. 健康窗口 + 触发器（不触发）
control monitor -ledger ledger.jsonl -workspace rcws -window-start 2026-08-20T12:00:00Z `
  -window-end 2026-08-20T14:00:00Z -thresholds mon_thresholds.json -triggers triggers.json `
  -out monitor_report.json -fired fired.jsonl
#   → 「overall=ok」「触发器: 0 条触发」

# 7. 发布状态机（人工批准默认；promote 请求打印 stdout 交给你的 adapter）
control release -state release_state.json -action candidate -release-id rel-demo `
  -evaluation evaluation.json -model rcws/models/model_rank_ndcg.leaves.json `
  -run-id run-demo -snapshot-id snap-demo -policy-version deal-v1 -last-known-good rel-0
control release -state release_state.json -action approve -approver demo-human
control release -state release_state.json -action confirm-promote -model-version m-demo
#   → stdout 打印 promote_request JSON（release_id/model_version/model_sha256）
control release -state release_state.json -action observe     # → observing，锚点 rel-0 不变

# 8. 退化注入 → 触发器 fired → 回滚指向 last_known_good
#    （deal 每用户只留 1 行 → deck_fill_rate=0.1 < 0.8 → block）
control monitor -ledger ledger.jsonl -workspace rcws -window-start 2026-08-20T12:00:00Z `
  -window-end 2026-08-20T15:00:00Z -thresholds mon_thresholds.json -triggers triggers.json `
  -out monitor_deg.jsonl -fired fired.jsonl
#   → fired.jsonl: {"rule":"deck-fill-hard","action":"rollback_requested","reason":"deck_fill_rate block ... value 0.1 ..."}
control release -state release_state.json -action rollback -reason "deck_fill_rate block（触发器 deck-fill-hard）"
#   → stdout 打印 rollback_request: {"from":"rel-demo","to":"rel-0",...}；state=rollback_requested

# 边界路径：deal 文件缺失/损坏 → 完整性 block（metric_not_computed）同样可触发回滚
```

## 12. 边界与非目标

- leaves 不托管：在线 serving、特征库、model registry、消息队列、在线学习。
- 事件真实采集/推送由应用侧实现；leaves 只定义可离线测试的契约与 adapter 接口。
- 无真实曝光事件时，离线 NDCG 不得推断为 CTR/CVR 收益。
- 初始版本人工批准默认开启；自动推广需应用侧具备签名、访问控制、监控与已验证回滚 adapter。
