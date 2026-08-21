# leaves v2.6.2

> **主题**：解析面模糊测试（含 1 个真实 panic 修复）· prep 时间切分主线 · SKILL 跨任务记忆  
> **日期**：2026-08-22

## Highlights

- **FUZZ 解析面模糊测试**：`io`（模型加载）/ `data`（格式嗅探）/ `recsys/contract`（事件校验）
  三个信任边界接入 Go 原生 fuzzing。**首轮 20 秒即抓到真实 panic**：
  `toitware/ubjson` 依赖解 4 字节畸形输入（`{"i\xef`）时负长度切片崩溃，
  已在 `parseXGBoostUBJSONBytes` 统一 recover 转错误；crash 语料入库作回归种子。
- **prep 时间切分主线（RC 遗留收口）**：`recsys.Interaction` 增可选 `Time`；
  `prep.RunTimeSplit` 提供 as-of 切分 + `CheckLeakage` 泄漏门禁；
  `SmokeConfig.SplitMode="time"` 一键切换四段流水线（默认用户切分行为不变）。
  synth 产确定性交织时间戳、MovieLens 读 `u.data` 真实时间戳；
  无时间戳诚实报错，不静默回退。
- **SKILL §4.6 跨任务记忆（lessons.md）**：演进方案 §16.2「远期观察」转正——
  把本任务的坑变成下任务的先验；核心闭环增「步骤 -1 读旧教训 / 步骤 8 沉淀教训」；
  库不读不写该文件，`runs.jsonl` 仍 CLI 独占。
- **控制面 CLI 修复（v2.6.1 遗留）**：`snapshot` 必填 `-time-start/-time-end`
  （usage 级 exit 1）；`docs/recsys-loop.md` §10.1 以实跑数字重写。

## Recommended API

- Infer: `LoadFromFile` + `DefaultLoadOptions`
- Train: `NewLearner` / `LoadDataAuto` / CLI `leaves train`
- Recsys 四段: `pipeline.Run(w, cfg)`（`cfg.SplitMode="time"` 可选时间切分）
- See `docs/api-surface.md`

## Compatibility

- Breaking: none
- `Interaction.Time` 为新增可选字段（零值=未知），samples TSV 四元格式不变
- 用户切分（默认）行为与产物完全不变；时间切分为 opt-in

## CI

- test（3 OS）/ lint / race / wasm / bench-gate
- 本地：`go test ./... -count=1` 全绿；fuzz 冒烟 io 30s / data 15s / contract 15s 全 PASS
- skills 镜像门禁绿（`TestSkillsMirrorSync`）

## Docs

- `docs/recsys-loop.md` §1 缺口表（时间切分已进四段主线）
- `skills/leaves-autotrain/SKILL.md` §4.6 + 核心闭环图（`.cursor/skills` 镜像同步）
- CHANGELOG Unreleased → [2.6.2]

## Fuzzing on demand

```powershell
go test ./io -fuzz FuzzLoadFromFileBytes -fuzztime 60s
go test ./data -fuzz FuzzSniffFileFormatBytes -fuzztime 60s
go test ./recsys/contract -fuzz FuzzValidateInteractionsJSON -fuzztime 60s
```
