# Changelog

本文件记录面向用户的版本变更。格式参考 [Keep a Changelog](https://keepachangelog.com/)。  
发版前勾选 [`docs/release-checklist.md`](docs/release-checklist.md)。

## [Unreleased]

### Added

- **sklearn pickle → ForestIR**（REV-01 收口）：`io.ParseSklearnPickleFile`；`LoadFromFile` 对实验 SK 路径也不再依赖根包 init。`SKEnsembleFromFile` 兼容入口保留；`TestSKIoLoadMatchesLegacy` 对照黄金预测。
- **`scripts/wgpu_repro`**：Windows 上 FromSlice+Gather+.Data 最小复现（REV-10 草稿指向此脚本）
- recsys `synth` / `tsvio` / `movielens` titles 包级单测

### Fixed

- **SHAP additivity vs LGB AutoTransform**：`explain.TestTreeSHAPAdditivity` 在 LGB 走 `DefaultLoadOptions`（现会套 logistic）时拿 `Predict` 当 margin；改为 `predict.OutputMargin`（SHAP 本就在 margin 空间）

### Changed — REV-09 godoc 与注释

- 根包 `doc.go` 示例改为 `LoadFromFile`；LightGBM JSON 兼容层读取 `average_output`
- 代码注释去掉过时 GoMLX 选型用语；`compatibility.md` 标明为 2019 历史表
- 删除恒返回 nil 的 `tree.LgTreeToTreeIR`；serving-template 间接依赖对齐 born v0.9.23

### Changed — REV-04…08 审阅落地

- **WASM Auto 一律 Native**（REV-04）：`GOOS=js` 上 `BornEngine` 委托 Native，作废 `wasm_born_cpu` / GPU-O3 1.6–2.6× 主张；规则码仅 `wasm_native`
- **`io.LoadFromFile` 不再需要根包 init**（REV-01/02）加载 LGB text/JSON、XGB JSON/UBJ/bin、leaves.json、ONNX、sklearn pickle
- **LGB 解析进 `io/` → ForestIR**；`LoadFromFile` 对 LGB 现遵循 `AutoTransform`（与 XGB/leaves.json 一致；raw 请设 `AutoTransform: false`）。`LGEnsembleFromFile` 仍为兼容入口（布尔参数语义不变）
- **`recsys/trainrank` 不再 import `demos/`**（REV-03）：评估工具迁 `recsys/rankutil`
- **backend-gate**（REV-05）：默认 `windows-latest` + `LEAVES_BORN_GPU=0`；self-hosted GPU 仅 `workflow_dispatch`+仓库变量
- **Born 守卫**（REV-08）：cat-small walk 回落标量而非 panic；显式 GPU Predict 8s 超时转错误
- deal/recall/rankconv 包级单测；io fuzz 增加 LGB 种子（REV-07）
- README/AGENTS **CLI 入口表**（REV-06）

### Changed — 文档与代码对齐

- **推荐 import**：示例从无法编译的 `github.com/linkerlin/leaves/v2/v2` 改为 `github.com/linkerlin/leaves/v2`；`PredictSingle` 示例改为真实签名（返回 `float64`，无 error）
- **ONNX**：interop-matrix / `SupportOf` / README 写上 TreeEnsembleClassifier 子集与 `LoadOnnxGraph`；不再声称 Classifier 不支持
- **BackendAuto 2.1**：英文 README 决策表与中文一致（桌面 Auto = Native）；`docs/bench/sample_benchrecords.jsonl` 规则码改为 `native_batch`
- **CLI**：api-surface / AGENTS 补 `lessons`/`version`；cli.md `leaves_cli` 改为真实版本标签；控制面 `snapshot` 用法补必填 `-time-start/-time-end`
- **门禁**：`TestReleaseDocsPresent` 改为 glob 全部 `release-notes-v*.md`；新增 `TestNoDoubleV2Import`

---

## [2.7.2] - 2026-08-22

> **主题**：GPU/Born 使用优化（GPU-O1..O6）——profiling 挂死守卫 · WASM 实测反转 · 叙事诚实化  
> **Release 正文**：[`docs/release-notes-v2.7.2.md`](docs/release-notes-v2.7.2.md)  
> **审计与路线**：[`GPU优化.md`](GPU优化.md)

### Fixed — profiling 挂死守卫（GPU-O1）

- **`tree.ProfileBackend` 三层守卫**：单后端 2s 计时预算（轮间检查，超预算 `Ok=false` 退出推荐）+ 均值 ns/op 下限 1e-3（防计时器归零——曾实测 BornGPU ≈0ns）+ `profileWithTimeout` 总超时 8s（goroutine+select；真挂起时 `profile_timeout` → Native）。**实测**：此前 `LEAVES_BACKEND_PROFILE=1` 下 batch≥256 GPU 挂死的 gate 脚本，现在 120s 内完整跑完。
- Windows 时钟粒度兜底：全部 iters 落在同一 tick 时按 1 tick 计（极小负载不再误判失败）。5 个守卫单测（`TestProfileBudgetGuard*` / `TestProfileWithTimeoutHang` / `TestProfileMinNsGuardLocked`）。

### Changed — WASM 决策表实测反转（GPU-O3）

- **WASM 上 BornCPU 是真的快（小批）**：node v24 wasm_exec 三轮实测（50 树×31 节点×30 特征），batch=8 BornCPU **快 1.6–2.6×**（wasm 解释器拖慢 Native 标量 walk，与桌面格局相反）；batch≥64 打平噪声区。
- 决策表 WASM 行拆分：`batch<64 且 Born 支持 → wasm_born_cpu`；`batch≥64 或不支持 → wasm_native`（新 rule；旧 `wasm_native_fallback` 移除）。`TestBackendAutoDecisionTable` 锁定。
- **`scripts/wasm_backend_bench`**：GOOS=js 常驻复测工具；实测表入 `docs/benchmark-baseline.md` §WASM 实测。

### Changed — 定位与叙事诚实化（GPU-O2/O4/O6）

- `tree.BornEngine`/`BornConfig.UseGPU` godoc：推理 Born = **parity/兼容路径**（张量 walk 每步回主机 + 树张量零缓存，实测 Native 的 0.03–0.16×）；BornGPU = **experimental**（参考机 wgpu v0.30.x 计时异常/挂起，自测后再上）。
- README 计算底座节改述：**「Born = 训练加速器 + ONNX 运行时；推理 golden 永远 Native（桌面端）」** + WASM 小批例外；训练加速节补 `LEAVES_BORN_GPU=0` 提示；backend-auto.md 部署表同步。

### Added — 治理（GPU-O5）

- **`.github/workflows/backend-gate.yml`**：每月 1 日复测 Native vs Born 双模型报数（防 born 升级缺口与决策表主张腐烂）。
- **`docs/upstream-wgpu-issue-draft.md`**：wgpu v0.30.x 计时归零/挂起/vkMapMemory panic 上游报告草稿（复现路径+疑点，待人工核对最小化后提交）。

---

## [2.7.1] - 2026-08-22

> **主题**：born v0.9.23 升级 + BackendAuto 2.1 诚实化（决策表「加速区」不可复现 → 默认 Native）  
> **Release 正文**：[`docs/release-notes-v2.7.1.md`](docs/release-notes-v2.7.1.md)

### Changed — BackendAuto 2.1（行为变更：Auto 不再按 batch 选 Born）

- **决策表诚实化**：`SelectBackendExplained` 删除 `born_gpu` / `born_cpu_gpu_unavailable` / `born_cpu` 三行，batch≥64 统一 `native_batch` → **Native**。依据：升级 born 时用官方 `tree.ProfileBackend` 计时路径复测 2.0「加速区」主张，**不可复现**——`lg_breast_cancer`（39 树/30 特征/849 节点）与合成森林（100×63 节点）上，born **v0.9.1 与 v0.9.23** 的 BornCPU 在 batch 64–4096 全段为 Native 的 **0.03–0.16×（慢 6–30×）**；BornGPU（wgpu v0.30.x）计时异常（≈0 或 batch≥256 挂起）。实测表入 [`docs/benchmark-baseline.md`](docs/benchmark-baseline.md) §再测量
- **走 Born 的两条路**（不受影响）：显式 `BackendBornCPU`/`BackendBornGPU`；`LEAVES_BACKEND_PROFILE=1` 实测选型（v2.7.0，测得更快才选）
- WASM 规则（`wasm_born_cpu`）保留（未参与本轮 CPU 实测，无证据翻转）
- `AutoBatchCPUThreshold`（64）语义改为 small_batch/native_batch 规则码分界；`AutoBatchGPUThreshold` Deprecated
- 测试与文档同步：`TestBackendAutoDecisionTable` 新表锁定；`docs/backend-auto.md` 2.1 决策表 + 变更说明；README 速查表；AGENTS.md；`io` 大批量路径测试更新

### Changed — 依赖

- **`github.com/born-ml/born` v0.9.1 → v0.9.23**（计算底座追平 22 个上游版本；编译零断裂，全量测试/parity/wasm 门禁绿；性能中性——见上表，两版本同量级）
- 传递依赖：`gogpu/wgpu` v0.29.0→v0.30.35、`go-webgpu/webgpu` v0.5.1→v0.5.5、`naga`/`goffi`/`gputypes`/`x/sys` 相应升级

### Added — 复测工具

- **`scripts/born_upgrade_gate`**：真实模型上 Native vs BornCPU/BornGPU 计时门禁（born 升级/决策表调整前必跑；`go run ./scripts/born_upgrade_gate <model>`），数字口径与 `tree.ProfileBackend` 一致

---

## [2.7.0] - 2026-08-22

> **主题**：ONNX Classifier 子集 · BackendAuto profiling 接线 · 独立 serving 产品仓 · lessons 可检索记忆库 · release 自动批准  
> **Release 正文**：[`docs/release-notes-v2.7.0.md`](docs/release-notes-v2.7.0.md)

### Added — ONNX TreeEnsembleClassifier 子集（ONNX-2）

- **`io/onnx_classifier.go`**：`ai.onnx.ml TreeEnsembleClassifier` 转 ForestIR——aggregate `SUM`/`AVERAGE`（类内树数折算）× post_transform `NONE`/`SOFTMAX`（n 类 n 组概率）/`LOGISTIC`（二类坍缩单输出组，p=sigmoid(raw)）；classlabels_int64s/strings 与按 max(class_id) 推断类数；base_values 按类展开
- **`parseONNXForest`**（`io/onnx.go`）：统一识别 Regressor/Classifier 节点；错误 hint 更新（提及 `LoadOnnxGraph` 通用图路径）。通用图推理（30+ 算子）仍走 `LoadOnnxGraph`（born 运行时，v2.4 已有）
- **`io.WriteONNXTreeEnsembleClassifier`**：最小分类器 proto 构造器（测试/样例）；原 Regressor 构造器内部助手重构为共享 `attr*` 写入函数
- 测试：SOFTMAX 手算对照、二类 LOGISTIC sigmoid、AVERAGE 折算、LOGISTIC+3 类可行动拒绝、回归器路径不回归

### Added — BackendAuto 二轮 profiling 接线（BAP-01）

- **`LEAVES_BACKEND_PROFILE=1|on|true|yes`**：`SelectBackendExplained` 在决策表之前对合成确定性批量实测 Native/BornCPU/BornGPU ns/op，选最快（Rule=`profile_*`，Reason 带实测数字与样本行数）
- 形状类缓存（batch × features × forest 规模）：每形状只测一次，不重复付首调用延迟；rows·cols ≤ 512k 成本封顶；`HasGPU=false` 时剔除 BornGPU 取次优
- **默认路径不变**：未设 env 时决策表 2.0 行为原样（`TestBackendProfileEnvOff/OffTruthy` 锁定）；opt-in API `tree.ProfileBackend`（v2.4）继续可用

### Added — 独立 serving 产品仓（SRV-01）

- **新仓库 [linkerlin/leaves-serving](https://github.com/linkerlin/leaves-serving)**：`examples/serving-template` 拆出为独立 module（require `leaves/v2 v2.6.2`、无 replace）、3-OS CI、Docker distroless、独立 README
- 仓内模板修复：go.mod require 原为 `github.com/linkerlin/leaves v0.0.0`（缺 `/v2`，独立构建坏）→ `leaves/v2 v2.6.2` + replace 对齐；README 指向独立仓
- 边界不变：仍非官方 serving 框架；鉴权/限流/registry 留给应用侧

### Added — lessons 可检索记忆库（LES2-01）

- **`leaves lessons add|search|list`**：跨任务记忆 CLI——存储 `~/.leaves/lessons.jsonl`（追加写 JSONL；`LEAVES_LESSONS_PATH` 可覆盖）；search 按词命中数降序（大小写不敏感，跨 task/lesson/evidence/tags）；输出单行 JSON（JSONL 语义）；库损坏行返回带修复 hint 的 `data_load` 错误
- **SKILL §4.6 升级双层记忆**：全局库（CLI）为主存储 + 项目 `lessons.md` 为可选人类可读镜像；核心闭环步骤 -1/8 改用 CLI；速查卡 9→10 命令；cli.md 新节；`.cursor/skills` 镜像同步
- 定位不变：CLI 只做存储/检索管道，「写什么何时读」策略仍在 SKILL/Agent

### Added — release 自动批准（RAU-01）

- **`recsys/release.AutoApprovePolicy`** + **`Machine.AutoApprove`**：candidate→approved 的策略化自动晋级——`Label` 必填（`ApprovedBy="auto:<Label>"` + 变迁 reason 带门禁统计，审计等价人工路径）；`RequireAllGatesPass`（有 warn 即拒）或 `MaxWarnGates`（显式放宽）
- 前提文档化：启用方进程须已具备签名/访问控制；状态机仍只产出 desired-state 请求（adapter 执行），无网络副作用；人工 `Approve` 仍为默认必经

### Included — 上轮 Unreleased（fuzz 深挖 + 真实数据时间切分）

- `recsys/ledger` fuzz 目标 `FuzzLedgerOpen`；定时 fuzz workflow（每周一 03:00 UTC，4 目标 60–90s）；`TestMovieLensTimeSplitFourStage` 真实 `u.data` 时间切分四段端到端；深挖战役全 PASS（`toitware/ubjson` 上游无修复可升，recover 防线保留）

---

## [2.6.2] - 2026-08-22

> **主题**：解析面模糊测试（含 1 个真实 panic 修复）· prep 时间切分主线 · SKILL 跨任务记忆  
> **Release 正文**：[`docs/release-notes-v2.6.2.md`](docs/release-notes-v2.6.2.md)

### Fixed — 信任边界加固（FUZZ 战果）

- **`io/xgb_ubjson.go`**：`toitware/ubjson` 依赖解 4 字节畸形输入（`{"i\xef`）时负长度切片 panic，无上游版本可升；`parseXGBoostUBJSONBytes` 信任边界统一 recover 转错误（所有 UBJSON 加载路径经过此处）。由 `FuzzLoadFromFileBytes` 首轮 20 秒发现；crash 语料留 `io/testdata/fuzz/` 作回归种子
- **控制面 CLI（v2.6.1 遗留）**：`snapshot` 子命令 `-time-start/-time-end` 改为必填（契约要求快照时间范围），usage 级失败（exit 1）而非运行时校验（exit 2）；`docs/recsys-loop.md` §10.1 以实跑数字重写

### Added — 解析面模糊测试（FUZZ）

- **`io/fuzz_test.go`** `FuzzLoadFromFileBytes`：任意字节经 `DetectFormat` + 遗留 loader + engine builder 不得 panic；种子含 testdata 小模型；外部测试包导入根包完成 loader 注册
- **`data/fuzz_test.go`** `FuzzSniffFileFormatBytes`：CSV/LIBSVM/RankingTSV/垃圾种子 → `SniffFileFormat` + `DetectFileFormat` 不 panic
- **`recsys/contract/fuzz_test.go`** `FuzzValidateInteractionsJSON`：任意 JSON 过 `ValidateInteractions` 不 panic，空 `event_id` 必被拒
- CI 默认以种子语料跑单测形态；深挖按需 `go test <pkg> -fuzz <名> -fuzztime 60s`（冒烟：io 30s 14k execs / data 15s / contract 15s 2.4M execs 全 PASS）

### Added — prep 时间切分主线（RC 遗留收口，演进方案 §17.1 P0）

- **`recsys.Interaction`** 增可选 `Time`（UTC；零值=未知；samples TSV 四元格式不变）
- **`recsys/synth`**：生成确定性交织时间戳（先按 seed 洗牌再赋单调 UTC 时间，避免 user-major 顺序使时间切分退化为按用户块切分）
- **`recsys/movielens`**：解析 `u.data` 第 4 列真实 Unix 时间戳
- **`recsys/split.SuggestTimeConfig`**：按 70/72/85 分位确定性推导边界（<10 事件拒绝）
- **`recsys/prep.RunTimeSplit`**：as-of 切分 + `CheckLeakage` 门禁 + 隔离带/val 丢弃计数；无时间戳诚实报错（不静默回退用户切分）；同一用户可重叠 train/test（QID 按 user+split 唯一，用户字典序保证确定性）；Tag 从 catalog 回填（`split.Assign` 只映射三元）
- **`recsys.SmokeConfig.SplitMode`**：`""`/`"user"`=用户切分（默认，行为不变）；`"time"`=时间切分（边界零值→自动推导）；pipeline 接线 + `TestPrepTimeSplitAuto` / `TestPrepTimeSplitRequiresTimestamps` / `TestSmokePipelineTimeSplit`
- `docs/recsys-loop.md` §1 缺口表更新：时间切分已进四段主线（opt-in）

### Added — SKILL 跨任务记忆（LESSONS，演进方案 §16.2 远期观察转正）

- **SKILL §4.6 跨任务记忆（lessons.md）**：与 runs.jsonl 同目录的 Markdown 表格；Agent 读写（库不读不写；runs.jsonl 仍 CLI 独占）；核心闭环增步骤 -1（读旧教训）与 8（沉淀教训）；写入仅限三处（反射假设证实/证伪、新失败模式、收敛后合并 ≤1 条并删证伪旧行）；镜像已同步

---

## [2.6.1] - 2026-08-21

> **主题**：控制面 CLI——八段剧本从 shell 驱动（Agent 零 Go 代码）  
> **Release 正文**：[`docs/release-notes-v2.6.1.md`](docs/release-notes-v2.6.1.md)

### Added — 控制面 CLI（recsys/cmd/control，2026-08-21）

- **`recsys/cmd/control`**：八段剧本的 shell 入口（Agent 零 Go 代码编排）——`snapshot`（工作区文件 sha256 + 特征指纹，`-time-start/-time-end` 必填：契约要求快照携带时间范围）、`split`（时间切分 + 泄漏检查）、`eval`（三层门禁）、`from-deal`（deal 终稿→决策账本）、`append-exposure/feedback`（事件摄取校验）、`replay`、`monitor`（阈值 + 触发器 → `fired.jsonl`）、`release`（状态机跨命令持久化，promote/rollback 请求打印 stdout）；退出码 0/1/2；端到端测试 `TestControlCLIEndToEnd` 跑完八段；`docs/recsys-loop.md` §10.1 实跑演练（2026-08-21 实测数字）。
- **`recsys/release`**：`MachineState` 导出/重构（`Export`/`FromState`）——CLI 跨命令恢复状态机的前提。
- **`recsys/eval`**：`RankViews` 从流水线工件构造评估视图（CLI/演练共用）。
- **`recsys/deal`**：`ReadLog` 读发牌日志 JSONL。
- 文档：`docs/recsys-loop.md` §10 CLI 指南；`recsys-orchestrator` SKILL §十 shell 版剧本（镜像同步）。

---

## [2.6.0] - 2026-08-21

> **主题**：推荐生产闭环控制面（演进方案 §十七 RC0–RC4）——七包契约 + 八段端到端演练  
> **Release 正文**：[`docs/release-notes-v2.6.0.md`](docs/release-notes-v2.6.0.md)

### Added — 推荐生产闭环控制面（演进方案 §十七 RC，2026-08-21）

- **`recsys/contract`**：冻结 schema v1 契约——`DatasetSnapshot`（输入文件 sha256 + `FeatureSchemaHash`，`VerifyFiles` 任一不符即失败）、`InteractionEvent`/`DecisionEvent`/`ExposureEvent`/`FeedbackEvent`（UTC 时间、匿名键绊线、原因码、回链校验：重复 ID / 未知类型 / 反向时间 / 位次不符均失败）、`ReleaseEvidence`；JSONL 泛型读写。
- **`recsys/split`**：时间切分（`train_end < val_start < test_start`）+ as-of 泄漏门禁 `CheckLeakage` + cold-start 分层；隔离带事件丢弃。
- **`recsys/eval`**：三层离线门禁（data / candidate_rank / deal）——Recall@K、coverage、NDCG/MAP、deck 质量（fill/dup/tag overflow/item coverage）、cold/returning 分层；阈值业务配置，缺失时仅 `exploratory`，未知指标 `block`；产出 `evaluation.json`。
- **`recsys/ledger`**：决策/曝光/反馈 JSONL 账本（append 即校验回链与时间因果）；`DecisionFromDeal` 把发牌终稿映射为审计真源决策（补位条目带 `tag_overflow` 原因码）。
- **`recsys/replay`**：归因窗样本重建——只有可归因到 shown 曝光的窗内反馈构成正样本；迟到/孤立/抑制反馈计数入 `replay_report.json`；确定性排序。
- **`recsys/monitor`**：窗口聚合（ctr / orphan_feedback_rate / deck 质量）→ `monitor_report.json`，状态 `ok|warn|block` + reason code；**触发器** `TriggerSet`——指标连续 N 窗口越界（warn/block）→ `retrain/rollback_requested`，带冷却期与恢复重置（§17.6「配置驱动，不是 Agent 临场猜测」）。
- **`recsys/release`**：发布状态机（`exploratory→candidate→approved→promoted→observing→retrain/rollback_requested/retired`）——evidence 三层门禁齐全 + 模型 hash 一致才可 candidate；人工批准默认；`last_known_good` 只指向已记录证据；`Adapter` 接口 + `FakeAdapter` 只产出推广/回滚请求（无网络副作用）；`run_status.jsonl`。
- **`recsys/loop`**：八段端到端演练 `TestAgenticRecsysLoopDrill`（快照/切分 → 门禁 evidence → fake adapter promoted → 账本 → 健康观察 → 退化注入 → 回滚指向 last_known_good → replay → retrain）。
- 文档：`docs/recsys-loop.md`（RC-00 基线 + 控制面指南）；`skills/recsys-orchestrator` §十八段剧本（镜像同步）；README / AGENTS / `examples/serving-template` 对接说明。

---

## [2.5.5] - 2026-08-16

> **主题**：module zip 瘦身——21.8 MB → ~5 MB  
> **Release 正文**：[`docs/release-notes-v2.5.5.md`](docs/release-notes-v2.5.5.md)

### Removed

- **`bin/install_pjrt.exe` / `bin/verify_pjrt.exe`**（合计 ~36 MB 原始体积，全仓零引用的孤儿实验二进制）：此前随 module zip 分发，`go get` 每次下载。module zip 由 git 追踪文件构成——**这是所有「不该提交的产物会被 go get 用户下载」问题的根因**。
- **`examples/wasm/leaves.wasm`**（3.5 MB，可重建产物；README/CI 均为现场构建，wasm 体积门禁也是临时目录新构建而非读该文件）。
- **`.chong/`**（11 个文件，另一 Agent 工具的工作记忆/事件日志，与库无关）。

以上均加 `.gitignore`；testdata（回归矩阵必需）与 logo 保留。

---

## [2.5.4] - 2026-08-16

> **主题**：manifest 复现契约修复——reproduce 不再丢路径语义  
> **Release 正文**：[`docs/release-notes-v2.5.4.md`](docs/release-notes-v2.5.4.md)

### Fixed

- **manifest.reproduce 丢参**：原构建器漏 `--cv`/`--max-leaves`/`--num-target`/`--val`/`--early-stop`——CV run 的复现命令会退化为全量单训（与记录的 `cv_mean` 不可比）；早停 run 复现丢 `--val --early-stop`。现按 run 类型补全（CV 行带 `--cv K`；早停行带 `--val X --early-stop N`，val 路径记录于 `params.val`，仅单跑路径记录）。
- **`manifest.leaves_cli`**：硬编码占位 `agentic-1` → 真实版本标签（`vX.Y.Z` 或 `(devel)+<短commit>`，同 `leaves version`）。

### Changed

- **复现语义分工**（写实既有行为）：`--from-run` 为定稿/变异起点，**不回填 val**（默认全量重训，SKILL 定稿流程不变）；忠实重放某次 run 用 `manifest.reproduce` 或显式 `--val`。

---

## [2.5.3] - 2026-08-16

> **主题**：`leaves version` 子命令  
> **Release 正文**：[`docs/release-notes-v2.5.3.md`](docs/release-notes-v2.5.3.md)

### Added

- **`leaves version`**：输出版本/构建信息 JSON（`{version, go[, commit]}`，基于 `debug.ReadBuildInfo`）。`go install github.com/linkerlin/leaves/v2/cmd/leaves@vX.Y.Z` 安装的用户与 Agent 可自查版本（原报「未知子命令」）；仓库内 `go build/run` 显示 `(devel)`（+ commit，若 VCS 信息被嵌入）。

---

## [2.5.2] - 2026-08-16

> **主题**：`--from-run` 谱系流程修复（tag 未命中不再硬错）  
> **Release 正文**：[`docs/release-notes-v2.5.2.md`](docs/release-notes-v2.5.2.md)

### Changed

- **`train --from-run --tag`（新 tag）回落最优行**：tag 不在账本时原报 `usage` 错误退出——SKILL §4.5 的「复现最优 + `--tag p:parent+mutation` 起新谱系名」流程无法直接组合。现回落最优行作父代（stderr 注明父代 tag，可审计），用户新 tag 原样写入本次账本行。与 `--from-run` 联用且要切换单跑路径时显式 `--cv 0`（父代 `cv_folds` 会连带 CV 路径，SKILL 已注明）。

---

## [2.5.1] - 2026-08-16

> **主题**：CI 修复（自 v2.2.0 起慢性红灯收口）  
> **Release 正文**：[`docs/release-notes-v2.5.1.md`](docs/release-notes-v2.5.1.md)

### Fixed

- **CI wasm 构建**：`examples/wasm` 带 `//go:build js`，构建步骤缺 `GOOS=js GOARCH=wasm` → "build constraints exclude all Go files"。
- **CI lint job**：golangci-lint-action v6 不支持 golangci-lint v2 → 升 v7；lint 以 `GOOS=windows` 分析（加速面 helper 仅被 `//go:build windows` 的 GPU 文件调用，linux 构建会误判 unused；lint 自 v2.1.6 起 action 崩溃从未跑完，升 v7 后才暴露）。
- **CI Windows GPU panic**：runner 只有 WARP 软件设备，`IsAvailable()` 探测通过但运行时 `DXGI_ERROR_DEVICE_REMOVED`。新增环境变量 **`LEAVES_BORN_GPU=0|off|false`**（Windows）强制关闭 WebGPU 路径（训练与推理全部回落 CPU）；CI test job 已设置。
- **CRLF 测试解析**：`model` 测试在 CI Windows（autocrlf）下解析 testdata 尾部 `\r` 失败（`"-0.670\r"`）→ 解析前 TrimSpace。
- **runs.jsonl `elapsed_ms` 契约收紧**：账本行 **必带** `elapsed_ms`（去掉 omitempty；`0` 表示 <1ms）——原实现 sub-ms 训练会丢字段，与「每行必带耗时」的账本语义不符。

---

## [2.5.0] - 2026-08-16

> **主题**：Agent 演化搜索（GEPA 对标）+ 用户侧 Agent 入口 + `/v2` 文档对齐  
> **Release 正文**：[`docs/release-notes-v2.5.0.md`](docs/release-notes-v2.5.0.md)

### Added

- **演化搜索账本信号（EVO-02）**：`train` 的 metrics.json 与 runs.jsonl 账本行新增 `n_trees`（模型树数）、`elapsed_ms`（训练耗时毫秒）；`--cv` 时账本行新增 `fold_metrics`（各折指标）。供 Agent 做「指标 vs 模型大小/耗时」权衡、折级 Pareto 选父与筛选晋级（见 `skills/leaves-autotrain/SKILL.md` §4.5 演化搜索协议与 [`演进方案.md`](演进方案.md) §十六）。未存模型的 CV 路径 `n_trees` 省略。
- **SKILL §4.5 演化搜索协议（EVO-01/04）**：Hall-of-Fame + 折级 Pareto 选父、反射式变异（ASI → 假设 → 定向变异）、交叉重组、`--cv 2` 筛选→`--cv 5` 晋级、预算帽 ≤15、谱系 tag 约定（`p:<父>+<变异>` / `x:<A>|<B>`）。
- **用户侧 Agent 入口**：README「三步让 Agent 帮你训练」快速上手；`CLAUDE.md` 适配器（Claude Code 自动读 AGENTS.md）；SKILL walkthrough 以实跑数字重写（演示 §4.5 全协议）。

### Fixed

- README/README.en 安装命令：`go install github.com/linkerlin/leaves@latest` → `go install github.com/linkerlin/leaves/v2/cmd/leaves@latest`（原漏 `/v2` 模块路径与 `cmd/leaves`，仓库外无法安装）。
- 全仓文档模块路径对齐 `/v2`：godoc 徽章/链接、import 示例（README/README.en/api-surface/extension-points/versioning/serving-template/recsys-rank）、NOTES §4 过时的 `go get` 建议、`testscripts/compatibility_*.py` 的 require/replace 与 import 模板。

---

## [2.4.1] - 2026-07-11

> **主题**：模块路径迁移 `github.com/linkerlin/leaves` → `github.com/linkerlin/leaves/v2`

### Changed

- **模块路径迁移**（commit cab6a6f）：`go.mod` 声明 `github.com/linkerlin/leaves/v2`；后续 tag 打在 `/v2` 路径下。迁移前旧 tag 仍挂在无后缀路径。

---

## [2.4.0] - 2026-07-11

> **主题**：BackendAuto 第二轮 — opt-in profiling 探测  
> **Release 正文**：[`docs/release-notes-v2.4.0.md`](docs/release-notes-v2.4.0.md)

### Added

- **BackendAuto 第二轮：opt-in profiling**（`tree.ProfileBackend`）：
  - `ProfileBackend(caps, vals, nrows, ncols, iters) ProfileResult`：warm-up + 计时 Native / BornCPU / BornGPU，返回各后端实测 ns/op 与最快推荐（`Pick/Rule/Reason`，Rule 码 `profile_*`）。
  - **不改 2.0 默认决策表**（`SelectBackendExplained` 行为不变）；opt-in：调用方传代表性批量样本得到测量证据后再决定 Backend。
  - 不支持的后端（cat-small 森林 / 无 WebGPU）`Ok=false` 不参与推荐；Native 始终计时。

---

## [2.3.0] - 2026-07-11

> **主题**：完整 ONNX Graph 导入（复用 born 运行时，原列「明确不做」）  
> **Release 正文**：[`docs/release-notes-v2.3.0.md`](docs/release-notes-v2.3.0.md)

### Added

- **完整 ONNX Graph 导入**（`io.LoadOnnxGraph`，原列「明确不做」，现落地）：
  - 复用 `github.com/born-ml/born/onnx` 运行时（30+ 算子，opset 1–21），不在 leaves 重实现算子。
  - `LoadOnnxGraph(data []byte) (OnnxModel, error)`：返回 `OnnxModel`，`Predict([]float32) ([]float32, error)` 单样本前向；暴露 InputNames/OutputNames/OpsetVersion。
  - 与既有 `LoadONNX`（TreeEnsembleRegressor 子集，wasm 可用）互补：通用 NN / 图推理走 born，仅非 wasm（wasm stub 返回可操作错误）。
  - 关键：born 公共 `onnx.Model` 接口的方法引用 `internal/tensor.RawTensor`，但 `born/tensor` 公共包以 `type RawTensor = tensor.RawTensor` 别名重导出，使 leaves 可构造张量并调用 Forward。

---

## [2.2.0] - 2026-07-11

> **主题**：向量叶 `multi_output_tree` 训练（multi-target 一棵树、叶子为向量）  
> **Release 正文**：[`docs/release-notes-v2.2.0.md`](docs/release-notes-v2.2.0.md)

### Added

- **向量叶 `multi_output_tree` 训练**（XGBoost `multi_output_tree`，原列「明确不做」，现落地）：
  - `train.Config.MultiOutputTree bool`：开启后每轮长**一棵** `OutputDim=numGroups` 的向量叶树（非 `one_output_per_tree` 的每类一棵）。
  - 分裂增益跨输出组求和（同一 `(feat,thr)` 共享）；叶权重为逐类 `-G_c/(H_c+λ)` 向量。
  - treebuilder 内部统一为 k 维（grad/hess `[n*k]`、`splitGain` 跨类求和、leaf-major `LeafValue` flatten）；k=1 退化为标量（现有路径零回归）。
  - 适用 `multi:*` 与 `NumTarget>1`；`NewLearner` 校验需多输出。
  - Born/WebGPU 增益扫描对 `k>1` 回退纯 CPU（正确，较慢；标量 k=1 仍享加速）。

---

## [2.1.6] - 2026-07-11

> **主题**：并发 DATA RACE 与 transformation panic 修复；golangci-lint v2 门禁全量落地  
> **Release 正文**：[`docs/release-notes-v2.1.6.md`](docs/release-notes-v2.1.6.md)

### Fixed

- **并发 DATA RACE**（`treebuilder.parallelEvalHistFeats`）：多 goroutine 共享同一 `row` buffer 写入 `Dense.Row()`；改为每 goroutine 独立 buffer。影响多线程 `hist` 训练的正确性
- **transformation panic**（`TransformType(Exponential).Name()` 数组越界）：`transformNames` 缺 `exponential` 项；`TransformExponential.Name()` 因此错报为 "logistic"。加载 LightGBM exponential objective 模型时触发

### Changed

- 新增 **golangci-lint v2** CI 门禁：`govet`（关 composites）/ `ineffassign` / `unused` / `staticcheck`（关 QF1003/ST1000/ST1003）/ `errcheck`（`std-error-handling` 预设 + 测试豁免）/ **gofmt** formatter，全部 blocking，0 issues 基线
- 新增 **race detector** CI job（`go test -race -short ./...`）
- 清除 **~360 行死代码**（unused linter 全清零）：explain interventional SHAP 簇、treebuilder 被 `batchAccumulateHistWebGPU` 取代的 GPU hist 旧路径（`accumulateHistWebGPU`×3 / `prebuildGPUHists` / `gpuHistBuildEnabled` / `gpuHistMinSamples`）、各处孤立 helper；永跳测试（`lgmsltr`/`lghiggs`/`xghiggs` fixture 未入库）、失效 GoMLX 脚本、误入库产物
- 修存量 **ineffassign / staticcheck / errcheck**：真实 error 忽略（transform / Predict / predictLeafIndicesInner / Seek / Iterate）改 `_ =` 显式丢弃；dead 赋值 / 空分支 / bool 比较 / nil-len 等逐项修正

### Added

- **测试覆盖**：`transformation`（0→6，含 `Name()` 越界回归）、`predict`（0→5，含 Engine 接口契约锁定）、`booster`（0→6，含 GBLinear zero-grad 端到端）；`internal/xgbin` 补 golden 节点断言（关闭遗留 TODO）
- **包文档** `doc.go`：`train` / `explain` / `objective` / `metrics` / `mat` / `util`（pkg.go.dev 不再裸符号）
- **文档门禁测试**：`TestNoDeadReadmeZhLinks`（扫所有根 .md 防死链复发）、`TestDocVersionRefsConsistent`（从文档头部提取版本号，校验所有引用一致）

### Documentation

- 项目状态 **维护期 → 开发期**；版本漂移修复（演进计划 v5.4 / 演进方案 v1.5 全局一致）；`doc.go` 拼写修复（implemetation / exibit / beacase）

---

## [2.1.5] - 2026-07-11

> **主题**：MovieLens 四段流水线（prep→召回→LTR→发牌）  
> **Release 正文**：[`docs/release-notes-v2.1.5.md`](docs/release-notes-v2.1.5.md)

### Added

- **MovieLens 四段流水线（DEMO-ML-4stage）**
  - `recsys/movielens`：纯 Go 加载 ml-100k → 四元 Dataset + 片名
  - `pipeline.RunFromDataset`：prep→召回→LTR→发牌（合成/真实数据共用）
  - `go run ./recsys/cmd/movielens` · `agent four-stage` · MCP `movielens_four_stage`
  - 召回策略：部分正样本 + 未交互补齐；发牌 `fillOverflow` 可凑满 DeckSize

### Documentation

- TUTORIAL §11.1、recsys-orchestrator / MovieLens SKILL、TODO 快照对齐

---

## [2.1.4] - 2026-07-11

> **主题**：MovieLens 推荐片名旁车 + walkthrough  
> **Release 正文**：[`docs/release-notes-v2.1.4.md`](docs/release-notes-v2.1.4.md)

### Added

- MovieLens 推荐旁车 `rank_movielens_*_meta.jsonl`（movie_id/title）；`agent recommend` / `cmd/recommend` 展示片名
- `demos/movielens/scripts/walkthrough.ps1` / `.sh` 端到端脚本
- `demos/movielens/rankutil/meta.go` 加载旁车

### Documentation

- TUTORIAL / README 同步 meta 与 walkthrough

---

## [2.1.3] - 2026-07-11

> **主题**：MovieLens Ranker Agent/MCP 全流程 demo 与教程  
> **Release 正文**：[`docs/release-notes-v2.1.3.md`](docs/release-notes-v2.1.3.md)

### Added

- **MovieLens Ranker Agent/MCP demo**
  - `demos/movielens/cmd/agent`：stdout JSON CLI（status/prepare/train/eval/recommend/full-pipeline）
  - `demos/movielens/cmd/mcp`：stdio MCP server（`movielens_*` tools）
  - `demos/movielens/agentops`：CLI/MCP 共用实现
  - 详尽教程 [`demos/movielens/TUTORIAL.md`](demos/movielens/TUTORIAL.md)
  - Agent SKILL [`skills/recsys-movielens-ranker/`](skills/recsys-movielens-ranker/SKILL.md)

### Documentation

- `recsys-orchestrator` / README / AGENTS 交叉链接 MovieLens Agent 路径

## [2.1.2] - 2026-07-11

> **主题**：Explain 性能缓存、可拆 serving 模板、SK/testdata 门禁  
> **Release 正文**：[`docs/release-notes-v2.1.2.md`](docs/release-notes-v2.1.2.md)

### Added

- **Serving 模板（LIB-30）**：[`examples/serving-template/`](examples/serving-template/) — 可整目录拆为独立仓（优雅退出、max batch、/ready、热加载演示、Dockerfile）；`examples/http` 仍为最小 embed
- **SK / testdata 门禁（LIB-11/12）**：sklearn 协议矩阵文档 + 失败用例；`TestTestdataMatrixArtifactsExist`

### Changed

- **Explain 性能（LIB-22）**：`TreeExplainer` 缓存树节点权重与背景 margin，多样本 SHAP 复用 path 缓冲（语义不变；`go test ./explain`）

### Documentation

- [benchmark-baseline.md](docs/benchmark-baseline.md) 增加 Tree SHAP bench 指引
- NOTES：模块路径与 `v2.x` tag 的代理提示

## [2.1.1] - 2026-07-11

> **主题**：v2.1 后加固 — Agent 契约门禁、ONNX TreeEnsemble 子集、多目标回归  
> **Release 正文**：[`docs/release-notes-v2.1.1.md`](docs/release-notes-v2.1.1.md)

### Added

- **CLI / Agent 契约加固**
  - `skills/` ↔ `.cursor/skills/` 镜像入库 + `TestSkillsMirrorSync` CI 门禁
  - `train --strict-flags` → `error=cv_conflict`（cv 与 val/early-stop 冲突）
  - `publish --print-repro`：stdout 打印复现 train 命令
  - `train --out-final`：早停时另存 final-round 侧车（`final_model` / `final_round`）
  - metrics.`train_accel`：Fit 后实际训练加速模式（与推理 BackendAuto 无关）
- **扩展示例** [`examples/extension/`](examples/extension/)：`custom:l1` + `max_abs_error`
- **Bench 样例工件** [`docs/bench/`](docs/bench/)：`BenchRecord` JSONL 格式对照
- **ONNX（实验）**：`TreeEnsembleRegressor` 极小子集导入（`BRANCH_LEQ`/`SUM`/`NONE`）
- **多目标训练（LIB-21）**：`data.MultiTarget` + `train.Config.NumTarget` + CLI `--num-target`（CSV 末 N 列标签；one_output_per_tree）
  - `predict` / `eval --num-target` 多目标闭环；[`examples/multitarget/`](examples/multitarget/)

### Changed

- cli.md：错误码表、data_quality 扫描上限（5000 行）、save-best「内存截断非重训」、registry 对接模板（S3/gh/OCI）
- [`docs/backend-auto.md`](docs/backend-auto.md)：训练 vs 推理交叉说明；第二轮候选边界
- ONNX 支持等级：占位 → **实验子集**（完整 Graph 仍不做）

### Documentation

- 演进方案 / TODO / interop-matrix / api-surface 同步

## [2.1.0] - 2026-07-10

> **主题**：Agentic 闭环契约收口 + 库线平台化（扩展点 / BackendAuto 2.0 / 互操作等级 / 发版治理）  
> **Release 正文**：[`docs/release-notes-v2.1.0.md`](docs/release-notes-v2.1.0.md)

### Added

- **Agentic CLI 闭环**（`cmd/leaves`）：`sniff|train|eval|predict|inspect|explain|publish`
  - 完备 `params` 账本、`--save-best` 默认、`--from-run` 复现
  - `--error-format json`、`manifest.reproduce`、`--emit-repro-script`
  - sniff `data_quality`、`--na-policy error|skip-row`
- **扩展点**：objective / metric 全量 `Register`；[`docs/extension-points.md`](docs/extension-points.md)
- **BackendAuto 2.0**：batch≥64→BornCPU、≥256+GPU→BornGPU、稀疏/小批 Native；[`docs/backend-auto.md`](docs/backend-auto.md)
- **互操作等级**：稳定 / 实验 / 占位；`*io.LoadError` 可操作 hint；[`docs/interop-matrix.md`](docs/interop-matrix.md)
- **发版治理**：[`docs/api-surface.md`](docs/api-surface.md)、[`docs/versioning.md`](docs/versioning.md)、[`docs/release-checklist.md`](docs/release-checklist.md)
- `tree.BenchRecord` 统一 bench JSONL；`SelectBackendExplained` 可解释选型

### Changed

- `DefaultLoadOptions` 默认 `AutoTransform: true`（见 [NOTES.md](NOTES.md)）
- BackendAuto：大 batch 无 GPU 时选 BornCPU（不再静默 Native）
- NOTES 瘦身为仍有效兼容注记；路线图 v5.4 / 演进方案 v1.4 收口

### Fixed

- sniff `n_features` 与 `feature_names` 一致
- 早停路径默认保存 best_round 模型（`model_round`）
- README 中英文互链（`README.md` ↔ `README.en.md`）

### Documentation

- skills/leaves-autotrain + Cursor 镜像；examples/autotrain walkthrough 刷新

### CI / 质量

- 本地 `go test ./... -count=1` 已通过（发版前建议再跑 CI 三 OS）
- 门禁：`docs` 文档存在性、CLI 契约、BackendAuto 决策表、io 支持等级

---

## 打 tag

```powershell
# 工作区已提交后：
git tag -a v2.1.0 -m "leaves v2.1.0: Agentic CLI + platform hardening"
git push origin v2.1.0
# GitHub Release 正文粘贴 docs/release-notes-v2.1.0.md
```

---

## 更早版本

历史能力基线（P0–T5 训练完备、Born 迁移、量化/WASM 等）见 git history 与 [`TODO.md`](TODO.md) 存档。
