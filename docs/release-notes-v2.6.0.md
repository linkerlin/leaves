# leaves v2.6.0

> **主题**：推荐生产闭环控制面（演进方案 §十七 RC0–RC4）  
> **日期**：2026-08-21

## Highlights

- **`recsys/` 控制面七包**（纯 Go、零运行时依赖、全部可离线测试）：数据快照/指纹（`contract`）、时间切分防泄漏（`split`）、三层离线门禁（`eval`）、决策/曝光/反馈账本（`ledger`）、归因回放（`replay`）、窗口监控 + 可配置触发器（`monitor`）、发布状态机 + adapter 请求（`release`）。
- **八段端到端演练** `recsys/loop/TestAgenticRecsysLoopDrill`：快照 → 门禁 evidence → fake adapter promoted → 账本 → 健康观察 → 退化注入 → 触发器回滚指向 last_known_good → replay → retrain。
- **触发器**：连续越界 + 冷却期 + 恢复重置的配置驱动规则（§17.6「不是 Agent 临场猜测」）。

## 边界（写实）

- 离线 `deal` 终稿 ≠ 线上推荐决策；`DecisionEvent` 才是审计真源。
- leaves 只产出 adapter-neutral 的推广/回滚**请求**（`release.Adapter` + `FakeAdapter`）；真实 registry/serving/CI 由应用仓库实现。
- 人工批准默认开启；无真实曝光事件时离线 NDCG 不得推断为 CTR/CVR 收益。
- 官方 registry / 在线 serving / 实时学习 / 分布式训练：继续**不做**。

## Recommended API

- 推理：`LoadFromFile` + `DefaultLoadOptions`；训练：`NewLearner` / `LoadDataAuto` / CLI `leaves train`
- 控制面（实验层，契约冻结 schema v1、字段只增不删）：`recsys/{contract,split,eval,ledger,replay,monitor,release}`；见 [docs/recsys-loop.md](recsys-loop.md) 与 [docs/api-surface.md](api-surface.md)

## Compatibility

- Breaking: none（全部为新增包 + 文档；`recsys` 四段流水线行为不变）
- `DecisionEvent.subject_id` 为新增可选字段（向后兼容）
- 演进方案文档升至 v2.2（引用已全局同步）

## CI

- test（3 OS）/ lint（golangci-lint v2.11.4，0 issues）/ race / wasm ≤16MiB / bench-gate
- 本地：`go test ./... -count=1` 全绿；`go test -race -short ./recsys/...` 绿；skills 镜像门禁绿

## Docs

- 新增 [docs/recsys-loop.md](recsys-loop.md)（RC-00 基线 + 控制面指南）
- `skills/recsys-orchestrator` §十八段剧本（`.cursor/skills` 镜像同步）
- README / README.en / AGENTS / api-surface / serving-template / MovieLens TUTORIAL 对接说明
