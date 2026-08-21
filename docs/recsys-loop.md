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

```powershell
go test ./recsys/... -count=1
```

## 10. 边界与非目标

- leaves 不托管：在线 serving、特征库、model registry、消息队列、在线学习。
- 事件真实采集/推送由应用侧实现；leaves 只定义可离线测试的契约与 adapter 接口。
- 无真实曝光事件时，离线 NDCG 不得推断为 CTR/CVR 收益。
- 初始版本人工批准默认开启；自动推广需应用侧具备签名、访问控制、监控与已验证回滚 adapter。
